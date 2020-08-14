import numpy as np
import grpc
import os
import cv2

from tensorrtserver.api import api_pb2
from tensorrtserver.api import grpc_service_pb2
from tensorrtserver.api import grpc_service_pb2_grpc
from tensorrtserver.api import InferContext
from tensorrtserver.api import ProtocolType

from utils.triton_utils import parse_model

if __name__ == '__main__':
    url = "localhost:8001"
    protocol = ProtocolType.from_str("grpc")
    model_name = "pinet_pytorch"
    model_version = 1

    parse_model(url, protocol, model_name)

    ctx = InferContext(url, protocol, model_name, 1, False, 0, False)

    test_image = cv2.imread("./tmp/image1.png")
    test_image = cv2.resize(test_image, (512, 256))
    test_image = test_image[:, :, ::-1]  # bgr to rgb
    test_image = test_image.astype(np.float32) / 255.0
    test_image = np.transpose(test_image, [2, 0, 1])  # HWC to CHW

    ctx.run({"INPUT__0": [test_image]},
            {"OUTPUT__0": InferContext.ResultFormat.RAW,
             "OUTPUT__1": InferContext.ResultFormat.RAW,
             "OUTPUT__2": InferContext.ResultFormat.RAW},
            batch_size=1)
