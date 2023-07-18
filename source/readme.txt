├── data 张量结构、数据加载
│   ├── load_data.cpp 
│   ├── tensor.cpp
│   └── tensor_utils.cpp
├── layer
│   ├── abstract 抽象算子结构(各算子的父类)、算子注册工厂
│   │   ├── layer.cpp
│   │   ├── layer_factory.cpp
│   │   └── param_layer.cpp
│   └── details 各个算子的实现
│       ├── adaptive_avgpooling.cpp
│       ├── adaptive_avgpooling.hpp
│       ├── batchnorm2d.cpp
│       ├── batchnorm2d.hpp
│       ├── cat.cpp
│       ├── cat.hpp
│       ├── convolution.cpp
│       ├── convolution.hpp
│       ├── expression.cpp
│       ├── expression.hpp
│       ├── flatten.cpp
│       ├── flatten.hpp
│       ├── hardsigmoid.cpp
│       ├── hardsigmoid.hpp
│       ├── hardswish.cpp
│       ├── hardswish.hpp
│       ├── linear.cpp
│       ├── linear.hpp
│       ├── maxpooling.cpp
│       ├── maxpooling.hpp
│       ├── platform.hpp
│       ├── relu.cpp
│       ├── relu.hpp
│       ├── sigmoid.cpp
│       ├── sigmoid.hpp
│       ├── silu.cpp
│       ├── silu.hpp
│       ├── softmax.cpp
│       ├── softmax.hpp
│       ├── sse_mathfun.hpp
│       ├── upsample.cpp
│       ├── upsample.hpp
│       ├── view.cpp
│       ├── view.hpp
│       ├── winograd.cpp
│       ├── winograd.hpp
│       ├── x86_usability.hpp
│       ├── yolo_detect.cpp
│       └── yolo_detect.hpp
├── parser 表达式层的解析
│   └── parse_expression.cpp
├── runtime 计算图的定义、运行时相关的流程
│   ├── ir.cpp 
│   ├── runtime_ir.cpp
│   ├── runtime_op.cpp
│   └── store_zip.cpp
└── utils 算子的运行时间统计
    └── time_logging.cpp
