import torch.nn as nn
import torch
from copy import deepcopy
import numpy as np
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from typing import Tuple, List, Union

# post processing
from instances.postprocessing_pinet import generate_result
from instances.postprocessing_pinet import eliminate_fewer_points
from instances.postprocessing_pinet import sort_along_y
from instances.postprocessing_pinet import eliminate_out
from instances.postprocessing_pinet import draw_points


class TrainerLaneDetector:
    def __init__(self, network, network_params, training_params):
        """
        Initialize
        """
        super(TrainerLaneDetector, self).__init__()

        self.network_params = network_params
        self.training_params = training_params

        self.lane_detection_network = network

        self.optimizer = torch.optim.Adam(self.lane_detection_network.parameters(),
                                          lr=training_params.l_rate,
                                          weight_decay=training_params.weight_decay)
        self.lr_scheduler = MultiStepLR(self.optimizer,
                                        milestones=training_params.scheduler_milestones,
                                        gamma=training_params.scheduler_gamma)
        self.current_epoch = 0
        self.current_step = 0
        self.current_loss = None

    def train(self, batch_sample, epoch, step):
        """ train
        :param
            inputs -> ndarray [#batch, 3, 256, 512]
            target_lanes -> [[4, 48],...,], len(List[ndarray]) = 8, ndarray -> [lanes=4, sample_pts=48]
            target_h -> [[4, 48],...,], len(List[ndarray]) = 8, ndarray -> [lanes=4, sample_pts=48]

        compute loss function and optimize
        """
        grid_x = self.network_params.grid_x
        grid_y = self.network_params.grid_y
        feature_size = self.network_params.feature_size

        real_batch_size = batch_sample["image"].shape[0]

        # generate ground truth
        ground_truth_point = batch_sample["detection_gt"]
        ground_truth_instance = batch_sample["instance_gt"]

        # convert numpy array to torch tensor
        ground_truth_point = torch.from_numpy(ground_truth_point).float()
        ground_truth_point = Variable(ground_truth_point).cuda()
        ground_truth_point.requires_grad = False

        ground_truth_instance = torch.from_numpy(ground_truth_instance).float()
        ground_truth_instance = Variable(ground_truth_instance).cuda()
        ground_truth_instance.requires_grad = False

        # inference lane_detection_network
        result = self.predict(batch_sample["image"])

        metrics = {}
        lane_detection_loss = 0
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
            confidance_gt = confidance_gt.view(real_batch_size, 1, grid_y, grid_x)  # [8,1,32,64]
            exist_condidence_loss = torch.sum((confidance_gt[confidance_gt == 1] - confidance[confidance_gt == 1]) ** 2) / torch.sum(confidance_gt == 1)

            # non exist confidance loss
            nonexist_confidence_loss = torch.sum((confidance_gt[confidance_gt == 0] - confidance[confidance_gt == 0]) ** 2) / torch.sum(confidance_gt == 0)

            # offset loss
            offset_x_gt = ground_truth_point[:, 1:2, :, :]
            offset_y_gt = ground_truth_point[:, 2:3, :, :]

            predict_x = offset[:, 0:1, :, :]
            predict_y = offset[:, 1:2, :, :]

            x_offset_loss = torch.sum((offset_x_gt[confidance_gt == 1] - predict_x[confidance_gt == 1]) ** 2) / torch.sum(confidance_gt == 1)
            y_offset_loss = torch.sum((offset_y_gt[confidance_gt == 1] - predict_y[confidance_gt == 1]) ** 2) / torch.sum(confidance_gt == 1)

            offset_loss = (x_offset_loss + y_offset_loss) / 2

            # compute loss for similarity
            sisc_loss = 0
            disc_loss = 0

            feature_map = feature.view(real_batch_size, feature_size, 1, grid_y * grid_x)  # [8, 4, 1, 2048]
            feature_map = feature_map.expand(real_batch_size, feature_size, grid_y * grid_x, grid_y * grid_x).detach()  # [8, 4, 2048, 2048]

            point_feature = feature.view(real_batch_size, feature_size, grid_y * grid_x, 1)  # [8, 4, 2048, 1]
            point_feature = point_feature.expand(real_batch_size, feature_size, grid_y * grid_x, grid_y * grid_x)  # .detach()  [8, 4, 2048, 2048]

            distance_map = (feature_map - point_feature) ** 2
            distance_map = torch.norm(distance_map, dim=1).view(real_batch_size, 1, grid_y * grid_x, grid_y * grid_x)

            # same instance
            sisc_loss = torch.sum(distance_map[ground_truth_instance == 1]) / torch.sum(ground_truth_instance == 1)

            # different instance, same class
            disc_loss = self.training_params.K1 - distance_map[ground_truth_instance == 2]  # self.p.K1/distance_map[ground_truth_instance==2] + (self.p.K1-distance_map[ground_truth_instance==2])
            disc_loss[disc_loss < 0] = 0
            disc_loss = torch.sum(disc_loss) / torch.sum(ground_truth_instance == 2)

            lane_loss = self.training_params.constant_exist * exist_condidence_loss + self.training_params.constant_nonexist * nonexist_confidence_loss + self.training_params.constant_offset * offset_loss
            instance_loss = self.training_params.constant_alpha * sisc_loss + self.training_params.constant_beta * disc_loss
            lane_detection_loss = lane_detection_loss + self.training_params.constant_lane_loss * lane_loss + self.training_params.constant_instance_loss * instance_loss

            metrics["hourglass_" + str(hourglass_id) + "_same_instance_same_class_loss"] = sisc_loss.item()
            metrics["hourglass_" + str(hourglass_id) + "_diff_instance_same_class_loss"] = disc_loss.item()
            metrics["hourglass_" + str(hourglass_id) + "_instance_loss"] = instance_loss.item()
            metrics["hourglass_" + str(hourglass_id) + "_confidence_loss"] = self.training_params.constant_exist * exist_condidence_loss.item() + self.training_params.constant_nonexist * nonexist_confidence_loss.item()
            metrics["hourglass_" + str(hourglass_id) + "_offset_loss"] = self.training_params.constant_offset * offset_loss.item()
            metrics["hourglass_" + str(hourglass_id) + "_total_loss"] = self.training_params.constant_lane_loss * lane_loss.item() + self.training_params.constant_instance_loss * instance_loss.item()

        metrics["pinet_total_loss"] = lane_detection_loss.item()

        self.optimizer.zero_grad()
        lane_detection_loss.backward()
        self.optimizer.step()

        del confidance, offset, feature
        del ground_truth_point, ground_truth_instance
        del feature_map, point_feature, distance_map
        del exist_condidence_loss, nonexist_confidence_loss, offset_loss, sisc_loss, disc_loss, lane_loss, instance_loss

        # update lr based on epoch
        if epoch != self.current_epoch:
            self.current_epoch = epoch
            self.lr_scheduler.step()

        if step != self.current_step:
            self.current_step = step

        self.current_loss = lane_detection_loss.item()

        return self.current_loss, metrics, result

    def predict(self, inputs: np.ndarray):
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

            confidence = confidences[i].view(self.network_params.grid_y, self.network_params.grid_x).cpu().data.numpy()

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

    def load_weights(self, path):
        self.lane_detection_network.load_state_dict(
            torch.load(path), False
        )

    def load_weights_v2(self, path):
        checkpoint = torch.load(path)
        self.lane_detection_network.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.current_step = checkpoint['step']
        self.current_loss = checkpoint['loss']

    def save_model(self, path):
        """ Save model """
        torch.save(
            self.lane_detection_network.state_dict(),
            path
        )

    def save_model_v2(self, path):
        """ Save model """
        torch.save({
            'epoch': self.current_epoch,
            'step': self.current_step,
            'model_state_dict': self.lane_detection_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.current_loss,
            }, path)

    def get_lr(self):
        return self.lr_scheduler.get_lr()

    @staticmethod
    def count_parameters(model: [nn.Module]):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)