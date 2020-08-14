# 1. NVidia Triton Inference Server 1.12 
* [official docs](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html)

### 1.1 Download Server Image
```
docker pull nvcr.io/nvidia/tritonserver:20.03-py3
```

### 1.2 Download Dummy Dataset
```
cd triton-inference-server/docs/examples
./fetch_models.sh
```

### 1.3 Start Server
* install **nvidia-docker** to enable GPU container
```
sudo docker run --gpus 1 --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 -v /home/zzhou/local_builds/triton-inference-server/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:20.03-py3 trtserver --model-store=/models
```

### 1.4 Model Repository and Folder Structure
* from offical doc
```
└── model_repository
    ├── densenet_onnx
    │   ├── 1
    │   │   └── model.onnx
    │   ├── config.pbtxt
    │   └── densenet_labels.txt
    ├── inception_graphdef
    │   ├── 1
    │   │   └── model.graphdef
    │   ├── config.pbtxt
    │   └── inception_labels.txt
    ├── resnet50_netdef
    │   ├── 1
    │   │   ├── init_model.netdef
    │   │   └── model.netdef
    │   ├── config.pbtxt
    │   └── resnet50_labels.txt
    ├── simple
    │   ├── 1
    │   │   └── model.graphdef
    │   └── config.pbtxt
    └── simple_string
        ├── 1
        │   └── model.graphdef
        └── config.pbtxt
```

* from my previsou examples
```
├── enet_pytorch
│   ├── 1
│   │   └── model.pt
│   ├── config.pbtxt
│   └── enet_labels.txt
└── ssd300_pytorch
    ├── 1
    │   └── model.pt
    ├── config.pbtxt
    └── ssd300_labels.txt
```

### 1.5 Model Configuration
* A minimal model configuration must specify 
    * name
    * platform
    * max_batch_size
    * input
    * output
    
* my enet example
```
name: "enet_pytorch"
platform: "pytorch_libtorch"
max_batch_size: 8
input [
  {
    name: "INPUT__0"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [ 3, 400, 960 ]
  }
]
output [
  {
    name: "OUTPUT__0"
    data_type: TYPE_FP32
    dims: [ 5, 400, 960]
    label_filename: "enet_labels.txt"
  }
]
```

* PyTorch Naming Convention: Due to the absence of names for inputs and outputs in the model, 
the “name” attribute of both the inputs and outputs in the configuration must follow 
a specific naming convention i.e. “<name>__<index>”. Where <name> can be any string and <index> 
refers to the position of the corresponding input/output. This means if there are two inputs 
and two outputs they must be name as: “INPUT__0”, “INPUT__1” and “OUTPUT__0”, “OUTPUT__1” 
such that “INPUT__0” refers to first input and INPUT__1 refers to the second input.

* if set **max_batch_size >= 1**, server will automatically assume batch channel like [x, 16]
* if set **max_batch_size = 0**, server will assume as original shape [16]
* **is_shape_tensor: true**, the server will assume as original shape [16], no matter **what max_batch_size** is