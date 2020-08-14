import numpy as np
import tensorrtserver.api.model_config_pb2 as model_config
from tensorrtserver.api import ServerStatusContext


def model_dtype_to_np(model_dtype):
    if model_dtype == model_config.TYPE_BOOL:
        return np.bool
    elif model_dtype == model_config.TYPE_INT8:
        return np.int8
    elif model_dtype == model_config.TYPE_INT16:
        return np.int16
    elif model_dtype == model_config.TYPE_INT32:
        return np.int32
    elif model_dtype == model_config.TYPE_INT64:
        return np.int64
    elif model_dtype == model_config.TYPE_UINT8:
        return np.uint8
    elif model_dtype == model_config.TYPE_UINT16:
        return np.uint16
    elif model_dtype == model_config.TYPE_FP16:
        return np.float16
    elif model_dtype == model_config.TYPE_FP32:
        return np.float32
    elif model_dtype == model_config.TYPE_FP64:
        return np.float64
    elif model_dtype == model_config.TYPE_STRING:
        return np.dtype(object)
    return None


def parse_model(url, protocol, model_name, verbose=False):
    """
    """

    ctx = ServerStatusContext(url, protocol, model_name, verbose)
    server_status = ctx.get_server_status()

    if model_name not in server_status.model_status:
        raise Exception("unable to get status for '" + model_name + "'")

    status = server_status.model_status[model_name]
    config = status.config

    # Model specifying maximum batch size of 0 indicates that batching
    # is not supported and so the input tensors do not expect an "N"
    # dimension (and 'batch_size' should be 1 so that only a single
    # image instance is inferred at a time).
    print(f"max_batch_size = {config.max_batch_size}")

    input_names = []
    for idx, each_input in enumerate(config.input):
        input_names.append(each_input.name)
        print("---------")
        print(f"input {idx}, name = {each_input.name}")
        print(f"input.dims = {each_input.dims}")
        print(f"input type = {model_config.DataType.Name(each_input.data_type)}")
        print(f"input format = {model_config.ModelInput.Format.Name(each_input.format)}")

    output_names = []
    for idx, each_output in enumerate(config.output):
        output_names.append(each_output.name)
        print("---------")
        print(f"output {idx}, name = {each_output.name}")
        print(f"output.dims = {each_output.dims}")
        print(f"output type = {model_config.DataType.Name(each_output.data_type)}")

    return input_names, output_names
