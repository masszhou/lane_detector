import torch.nn as nn
import torch
from copy import deepcopy
import numpy as np
from torch.autograd import Variable
import time
from typing import List, Tuple

from model.net import PINetJIT
# post processing
from instances.postprocessing_pinet import generate_result
from instances.postprocessing_pinet import generate_result_jit
from instances.postprocessing_pinet import eliminate_fewer_points
from instances.postprocessing_pinet import eliminate_fewer_points_jit
from instances.postprocessing_pinet import sort_along_y
from instances.postprocessing_pinet import eliminate_out
from instances.postprocessing_pinet import draw_points


class LaneDetector:
    def __init__(self, model_path, parameter):
        """
        Initialize
        """
        super(LaneDetector, self).__init__()

        self.p = parameter
        net = PINetJIT()
        net.load_state_dict(
            torch.load(model_path), False
        )
        net.cuda()
        net.eval()
        # quote from https://pytorch.org/tutorials/beginner/deploy_seq2seq_hybrid_frontend_tutorial.html#
        # you must call .to(device) to set the device options of the models
        # and .eval() to set the dropout layers to test mode before tracing the models.
        self.network_jit = torch.jit.trace(net, (torch.rand(1, 3, 256, 512).cuda(),))

    def count_parameters(self, model: [nn.Module]):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def predict(self, inputs: np.ndarray) -> torch.tensor:
        """
        predict lanes

        :param inputs -> [batch_size, 3, 256, 512]
        :return:
        """
        inputs = torch.from_numpy(inputs).float()
        inputs = Variable(inputs).cuda()
        return self.network_jit(inputs)

    def test_on_image(self, test_images, thresh=None):
        """ predict, then post-process

        """
        if thresh is None:
            thresh = self.p.threshold_confidence
        rank = len(test_images.shape)
        if rank == 3:
            batch_image = np.expand_dims(test_images, 0)
        elif rank == 4:
            batch_image = test_images
        else:
            raise IndexError

        start = time.time()
        result = self.predict(batch_image)  # accept rank = 4 only
        end = time.time()
        print(f"predict time: {end - start} [sec]")  # [second]

        confidences, offsets, instances = result  # use output of last hourglass block
        # confidences -> [1, 1, 32, 64] tensor
        # offsets -> [1, 2, 32, 64] tensor
        # instances -> [1, 4, 32, 64] tensor

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

            confidence = confidences[i].view(self.p.grid_y, self.p.grid_x).cpu().data.numpy()
            # [1, 32, 64] -> [32, 64]

            offset = offsets[i].cpu().data.numpy()
            offset = np.rollaxis(offset, axis=2, start=0)
            offset = np.rollaxis(offset, axis=2, start=0)
            # [1, 32, 64] -> [32, 64, 2]

            instance = instances[i].cpu().data.numpy()
            instance = np.rollaxis(instance, axis=2, start=0)
            instance = np.rollaxis(instance, axis=2, start=0)
            # [4, 32, 64] -> [32, 64, 4]

            # generate point and cluster
            raw_x, raw_y = generate_result(confidence, offset, instance, thresh)

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

