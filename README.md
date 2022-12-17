# KuiperInfer
自制的一款推理框架，项目整体风格和结构借鉴了Caffe，仅用作学习参考。

## 使用的技术
开发语言：C++ 17

数学库：Armadillo+OpenBlas

加速库：OpenMP

单元测试：GTest

性能测试：Google Benchmark

## 已经支持的算子
**总体秉承算子用到再开发的理念；**
**对已经实现的算子不做过度设计，用到再进行匹配**

1. Convolution 
2. AdaptivePooling 
3. MaxPooling 
4. Expression(抽象语法树)
5. Flatten, View(只支持HW维度展平和变形,其他维度用到再开发)
6. Sigmoid 
7. HardSigmoid 
8. HardSwish 
9. Relu 
10. Linear(矩阵相乘，只支持二维Tensor相乘，其他维度用到再开发)
11. Softmax 
12. BatchNorm

## 目录
1. source是源码目录
    * data/ 是张量类Tensor的实现和Tensor初始化方法
    * layer/ 是算子的实现
    * parser/ 是Pnnx表达式的解析类
    * runtime/ 是计算图结构，解析和运行时相关
2. test是单元测试目录，由于是个人项目，不能做到单元测试全覆盖。
3. bench是google benchmark, 包含对MobilenetV3和Resnet18的性能测试。


## 性能
### 测试环境
Ubuntu 22.04, Intel(R) Core(TM) i7-11800H @ 2.30GHz, 32G Memory
### 输入大小
8x3x224x224, NCHW格式
### 模型运行速度


|  Benchmark  | Batch Size | Time          | CPU        |Iterations|
|  ----  |------------|---------------|---------------|---------------|
| BM_Resnet18  | 8          | 1410739345 ns |738977365 ns|5|
| BM_MobilenetV3 | 8          | 1192045921 ns |726926230 ns|5|

## Flag
在b站上开一门教学课程，欢迎大家关注。