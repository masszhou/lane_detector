import torch.nn as nn
import torch
from copy import deepcopy
import numpy as np
from torch.autograd import Variable
from collections import OrderedDict
from typing import Tuple, List, Optional

# post processing
from instances.postprocessing_pinet import generate_result
from instances.postprocessing_pinet import eliminate_fewer_points
from instances.postprocessing_pinet import sort_along_y
from instances.postprocessing_pinet import eliminate_out
from instances.postprocessing_pinet import draw_points
from instances.postprocessing_pinet import sort_batch_along_y


class TrainerLaneDetector:
    def __init__(self, network, params):
        """
        Initialize
        """
        super(TrainerLaneDetector, self).__init__()

        self.params = params
        self.lane_detection_network = network
        self.setup_optimizer()
        self.current_epoch = 0
        self.train_log = OrderedDict()  # (step, metrics)

    def count_parameters(self, model: [nn.Module]):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def setup_optimizer(self):
        self.lane_detection_optim = torch.optim.Adam(self.lane_detection_network.parameters(),
                                                     lr=self.params.l_rate,
                                                     weight_decay=self.params.weight_decay)

    def make_ground_truth_keypoints(self, target_lanes: List[np.ndarray], target_h: List[np.ndarray]) -> Optional[np.ndarray, None]:
        """ Make ground truth for key point estimation
        0. sort GT lane points
        1. initialize a keypoints grid
        2. calculate offset for each labeled points, w.r.t. keypoints grid
        3. assign confidence/class to anchor point from keypoints grid
        ground[lane_id][0] -> confidence
        ground[lane_id][1:3] -> anchor for gt offset

        :param
            target_lanes -> List[ndarray] e.g. [[4, 48],[2, 48],[3, 48]...,]
            target_h -> List[ndarray] e.g. [[4, 48],[2, 48],[3, 48],...,]
        :return
            gt_detection -> ndarray [#lane, 3, grid_y, grid_x]
        """
        target_lanes, target_h = sort_batch_along_y(target_lanes, target_h)

        ground = np.zeros((len(target_lanes), 3, self.params.grid_y, self.params.grid_x))  # e.g. [#lane, 3, 32, 64]

        for batch_index, batch in enumerate(target_lanes):
            for lane_index, lane in enumerate(batch):
                for point_index, point in enumerate(lane):
                    if point > 0:
                        x_index = int(point / self.params.resize_ratio)  # resize ratio = 8, resize [256, 512] to [32, 64]
                        y_index = int(target_h[batch_index][lane_index][point_index] / self.params.resize_ratio)
                        ground[batch_index][0][y_index][x_index] = 1.0  # confidence
                        ground[batch_index][1][y_index][x_index] = (
                                                                               point * 1.0 / self.params.resize_ratio) - x_index  # offset x
                        ground[batch_index][2][y_index][x_index] = (target_h[batch_index][lane_index][
                                                                        point_index] * 1.0 / self.params.resize_ratio) - y_index  # offset y

        return ground

    def make_ground_truth_instance(self, target_lanes, target_h):
        """ Make ground truth for instance feature

        for Similarity Group Proposal Network(SPGN) loss, to cluster different lanes

        :param
            target_lanes -> List[ndarray] e.g. [[4, 48],[2, 48],[3, 52]...,]
            target_h -> List[ndarray] e.g. [[4, 48],[2, 48],[3, 52],...,]
        :return
            gt_instance-> ndarray [8, 1, grid_x * grid_y, grid_x * grid_y]
        """
        ground = np.zeros(
            (len(target_lanes), 1, self.params.grid_y * self.params.grid_x, self.params.grid_y * self.params.grid_x))  # [8, 1, 2048, 2048]
        # e.g. the grid is 32x64, thus there are 2048 anchor points
        # the similarity matrix is 2048x2048

        for batch_index, batch in enumerate(target_lanes):
            # batch -> ndarray e.g. [3, 52]
            temp = np.zeros((1, self.params.grid_y, self.params.grid_x))  # e.g. [1, 32, 64]
            lane_cluster = 1
            for lane_index, lane in enumerate(batch):
                previous_x_index = 0
                previous_y_index = 0
                for point_index, point in enumerate(lane):
                    if point > 0:
                        x_index = int(point / self.params.resize_ratio)
                        y_index = int(target_h[batch_index][lane_index][point_index] / self.params.resize_ratio)
                        temp[0][y_index][x_index] = lane_cluster
                    if previous_x_index != 0 or previous_y_index != 0:  # interpolation make more dense data
                        temp_x = previous_x_index
                        temp_y = previous_y_index
                        while True:
                            delta_x = 0
                            delta_y = 0
                            temp[0][temp_y][temp_x] = lane_cluster
                            if temp_x < x_index:
                                temp[0][temp_y][temp_x + 1] = lane_cluster
                                delta_x = 1
                            elif temp_x > x_index:
                                temp[0][temp_y][temp_x - 1] = lane_cluster
                                delta_x = -1
                            if temp_y < y_index:
                                temp[0][temp_y + 1][temp_x] = lane_cluster
                                delta_y = 1
                            elif temp_y > y_index:
                                temp[0][temp_y - 1][temp_x] = lane_cluster
                                delta_y = -1
                            temp_x += delta_x
                            temp_y += delta_y
                            if temp_x == x_index and temp_y == y_index:
                                break
                    if point > 0:
                        previous_x_index = x_index
                        previous_y_index = y_index
                lane_cluster += 1

            for i in range(self.params.grid_y * self.params.grid_x):  # make gt
                temp = temp[temp > -1]
                gt_one = deepcopy(temp)
                if temp[i] > 0:
                    gt_one[temp == temp[i]] = 1  # same instance
                    if temp[i] == 0:
                        gt_one[temp != temp[i]] = 3  # different instance, different class
                    else:
                        gt_one[temp != temp[i]] = 2  # different instance, same class
                        gt_one[temp == 0] = 3  # different instance, different class
                    ground[batch_index][0][i] += gt_one

        return ground

    def train(self, inputs, target_lanes, target_h, epoch, step):
        """ train
        :param
            inputs -> ndarray [#batch, 3, 256, 512]
            target_lanes -> [[4, 48],...,], len(List[ndarray]) = 8, ndarray -> [lanes=4, sample_pts=48]
            target_h -> [[4, 48],...,], len(List[ndarray]) = 8, ndarray -> [lanes=4, sample_pts=48]

        compute loss function and optimize
        """
        real_batch_size = len(target_lanes)

        # generate ground truth
        ground_truth_point = self.make_ground_truth_keypoints(target_lanes, target_h)
        ground_truth_instance = self.make_ground_truth_instance(target_lanes, target_h)

        # convert numpy array to torch tensor
        ground_truth_point = torch.from_numpy(ground_truth_point).float()
        ground_truth_point = Variable(ground_truth_point).cuda()
        ground_truth_point.requires_grad = False

        ground_truth_instance = torch.from_numpy(ground_truth_instance).float()
        ground_truth_instance = Variable(ground_truth_instance).cuda()
        ground_truth_instance.requires_grad = False

        # update lane_detection_network
        result = self.predict(inputs)
        lane_detection_loss = 0

        metrics = {}
        for hourglass_id, intermediate_loss in enumerate(result):
            confidance, offset, feature = intermediate_loss
            # e.g.
            # confidence shape = [8, 1, 32, 64]
            # offset shape = [8, 3, 32, 64]
            # feature shape = [8, 4, 32, 64]  for instance segmentation
            # quote
            # "The feature size is set to 4, and this size is observed to have no major effect for the performance."

            # compute loss for point prediction
            offset_loss = 0
            exist_condidence_loss = 0
            nonexist_confidence_loss = 0

            # exist confidance loss
            confidance_gt = ground_truth_point[:, 0, :, :]  # [8,1,32,64]
            confidance_gt = confidance_gt.view(real_batch_size, 1, self.params.grid_y, self.params.grid_x)  # [8,1,32,64]
            exist_condidence_loss = torch.sum(
                (confidance_gt[confidance_gt == 1] - confidance[confidance_gt == 1]) ** 2) / torch.sum(
                confidance_gt == 1)

            # non exist confidance loss
            nonexist_confidence_loss = torch.sum(
                (confidance_gt[confidance_gt == 0] - confidance[confidance_gt == 0]) ** 2) / torch.sum(
                confidance_gt == 0)

            # offset loss
            offset_x_gt = ground_truth_point[:, 1:2, :, :]
            offset_y_gt = ground_truth_point[:, 2:3, :, :]

            predict_x = offset[:, 0:1, :, :]
            predict_y = offset[:, 1:2, :, :]

            x_offset_loss = torch.sum(
                (offset_x_gt[confidance_gt == 1] - predict_x[confidance_gt == 1]) ** 2) / torch.sum(confidance_gt == 1)
            y_offset_loss = torch.sum(
                (offset_y_gt[confidance_gt == 1] - predict_y[confidance_gt == 1]) ** 2) / torch.sum(confidance_gt == 1)

            offset_loss = (x_offset_loss + y_offset_loss) / 2

            # compute loss for similarity
            sisc_loss = 0
            disc_loss = 0

            feature_map = feature.view(real_batch_size, self.params.feature_size, 1, self.params.grid_y * self.params.grid_x)  # [8, 4, 1, 2048]
            feature_map = feature_map.expand(real_batch_size, self.params.feature_size, self.params.grid_y * self.params.grid_x,
                                             self.params.grid_y * self.params.grid_x).detach()  # [8, 4, 2048, 2048]

            point_feature = feature.view(real_batch_size, self.params.feature_size, self.params.grid_y * self.params.grid_x, 1)  # [8, 4, 2048, 1]
            point_feature = point_feature.expand(real_batch_size, self.params.feature_size, self.params.grid_y * self.params.grid_x,
                                                 self.params.grid_y * self.params.grid_x)  # .detach()  [8, 4, 2048, 2048]

            distance_map = (feature_map - point_feature) ** 2
            distance_map = torch.norm(distance_map, dim=1).view(real_batch_size, 1, self.params.grid_y * self.params.grid_x,
                                                                self.params.grid_y * self.params.grid_x)

            # same instance
            sisc_loss = torch.sum(distance_map[ground_truth_instance == 1]) / torch.sum(ground_truth_instance == 1)

            # different instance, same class
            disc_loss = self.params.K1 - distance_map[ground_truth_instance == 2]  # self.p.K1/distance_map[ground_truth_instance==2] + (self.p.K1-distance_map[ground_truth_instance==2])
            disc_loss[disc_loss < 0] = 0
            disc_loss = torch.sum(disc_loss) / torch.sum(ground_truth_instance == 2)

            lane_loss = self.params.constant_exist * exist_condidence_loss + self.params.constant_nonexist * nonexist_confidence_loss + self.params.constant_offset * offset_loss
            instance_loss = self.params.constant_alpha * sisc_loss + self.params.constant_beta * disc_loss
            lane_detection_loss = lane_detection_loss + self.params.constant_lane_loss * lane_loss + self.params.constant_instance_loss * instance_loss

            metrics["hourglass_" + str(hourglass_id) + "_same_instance_same_class_loss"] = sisc_loss.item()
            metrics["hourglass_" + str(hourglass_id) + "_diff_instance_same_class_loss"] = disc_loss.item()
            metrics["hourglass_" + str(hourglass_id) + "_instance_loss"] = instance_loss.item()
            metrics["hourglass_" + str(hourglass_id) + "_confidence_loss"] = self.params.constant_exist * exist_condidence_loss.item() + self.params.constant_nonexist * nonexist_confidence_loss.item()
            metrics["hourglass_" + str(hourglass_id) + "_offset_loss"] = self.params.constant_offset * offset_loss.item()
            metrics["hourglass_" + str(hourglass_id) + "_total_loss"] = self.params.constant_lane_loss * lane_loss.item() + self.params.constant_instance_loss * instance_loss.item()

        metrics["pinet_total_loss"] = lane_detection_loss.item()
        self.train_log[step] = metrics

        self.lane_detection_optim.zero_grad()
        lane_detection_loss.backward()
        self.lane_detection_optim.step()

        del confidance, offset, feature
        del ground_truth_point, ground_truth_instance
        del feature_map, point_feature, distance_map
        del exist_condidence_loss, nonexist_confidence_loss, offset_loss, sisc_loss, disc_loss, lane_loss, instance_loss

        # quick fix, change loss weight after warmup
        if epoch > 0 and epoch % 20 == 0 and self.current_epoch != epoch:
            self.current_epoch = epoch
            if epoch > 0 and (epoch == 1000):
                self.params.constant_lane_loss += 0.5
                self.params.constant_nonexist += 0.5
                self.params.l_rate /= 2.0
                self.setup_optimizer()

        return lane_detection_loss.item()

    def predict(self, inputs: np.ndarray) -> torch.tensor:
        """
        predict lanes

        :param inputs -> [batch_size, 3, 256, 512]
        :return:
        """
        inputs = torch.from_numpy(inputs).float()
        inputs = Variable(inputs).cuda()
        return self.lane_detection_network(inputs)

    def test_on_image(self, test_images: np.ndarray,
                      threshold_confidence: float = 0.81) -> Tuple[List[List[int]], List[List[int]], List[np.ndarray]]:
        """ predict, then post-process

        :param test_images: input image or image batch
        :param threshold_confidence, if confidence of detected key points greater than threshold, then will be accepted
        """
        rank = len(test_images.shape)
        if rank == 3:
            batch_image = np.expand_dims(test_images, 0)
        elif rank == 4:
            batch_image = test_images
        else:
            raise IndexError

        # start = time.time()
        result = self.predict(batch_image)  # accept rank = 4 only
        # end = time.time()
        # print(f"predict time: {end - start} [sec]")  # [second]

        confidences, offsets, instances = result[-1]  # trainer use different output, compared with inference

        num_batch = batch_image.shape[0]

        out_x = []
        out_y = []
        out_images = []

        for i in range(num_batch):
            # test on test data set
            image = deepcopy(batch_image[i])
            image = np.rollaxis(image, axis=2, start=0)
            image = np.rollaxis(image, axis=2, start=0) * 255.0
            image = image.astype(np.uint8).copy()

            confidence = confidences[i].view(self.params.grid_y, self.params.grid_x).cpu().data.numpy()

            offset = offsets[i].cpu().data.numpy()
            offset = np.rollaxis(offset, axis=2, start=0)
            offset = np.rollaxis(offset, axis=2, start=0)

            instance = instances[i].cpu().data.numpy()
            instance = np.rollaxis(instance, axis=2, start=0)
            instance = np.rollaxis(instance, axis=2, start=0)

            # generate point and cluster
            raw_x, raw_y = generate_result(confidence, offset, instance, threshold_confidence)

            # eliminate fewer points
            in_x, in_y = eliminate_fewer_points(raw_x, raw_y)

            # sort points along y
            in_x, in_y = sort_along_y(in_x, in_y)
            in_x, in_y = eliminate_out(in_x, in_y)
            in_x, in_y = sort_along_y(in_x, in_y)
            in_x, in_y = eliminate_fewer_points(in_x, in_y)

            result_image = draw_points(in_x, in_y, deepcopy(image))

            out_x.append(in_x)
            out_y.append(in_y)
            out_images.append(result_image)

        return out_x, out_y, out_images

    def training_mode(self):
        """ Training mode """
        self.lane_detection_network.train()

    def evaluate_mode(self):
        """ evaluate(test mode) """
        self.lane_detection_network.eval()

    def cuda(self):
        """ Setup GPU computation """
        self.lane_detection_network.cuda()

    def load_weights(self, epoch, loss):
        self.lane_detection_network.load_state_dict(
            torch.load(self.params.model_path + str(epoch) + '_' + str(loss) + '_lane_detection_network.pkl'), False
        )

    def load_weights_v2(self, path):
        self.lane_detection_network.load_state_dict(
            torch.load(path), False
        )

    def save_model(self, epoch, loss):
        """ Save model """
        file_name = self.params.save_path + str(epoch) + "_" + str(loss) + "_lane_detection_network.pkl"
        torch.save(
            self.lane_detection_network.state_dict(),
            file_name
        )

    def save_model_v2(self, path):
        """ Save model """
        torch.save(
            self.lane_detection_network.state_dict(),
            path
        )

    def save_train_log(self):
        import json
        file_name = self.params.save_path + "lane_detection_network.json"
        with open(file_name, 'a') as f:
            f.write(json.dumps(self.train_log))