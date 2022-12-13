# KuiperInfer
自制的一款推理框架，项目整体风格和结构借鉴了Caffe. 数学运算使用`armadillo`实现，目前已支持resnet18的推理。
## 目录
1. source是源码目录
    * data/ 是张量类的实现
    * layer/ 是算子的实现
    * parser/ 是Pnnx表达式的解析类
    * runtime/ 是计算图解析和运行时相关
2. test是单元测试目录，由于是个人项目，不能做到单元测试全覆盖。
3. bench是google benchmark, 包含对MobilenetV3和Resnet18的性能测试。


## 性能
### 测试环境
Ubuntu 22.04, Intel(R) Core(TM) i7-11800H @ 2.30GHz, 32G Memory
### 输入大小
8x3x224x224, NCHW格式
### 模型运行速度
MobilenetV3Small:  130ms per batch

Resnet18: 180ms per batch
## Flag
在b站上开一门教学课程，欢迎大家关注