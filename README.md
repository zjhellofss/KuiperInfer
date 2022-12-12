# KuiperInfer
自制的一款推理框架，项目整体风格和结构借鉴了Caffe. 数学运算使用`armadillo`实现，目前已支持resnet18的推理。

对于`[3x224x224]`的输入在`i7-11800H`的运行速度大致为260ms，Convolution仅采用了`im2col`方式，仍有较大的优化空间。
1. source下是源码目录，分别是layer算子实现，runtime运行时和data张量类。
2. test是单元测试目录，但是个人精力有限，不能做单元测试全覆盖。
3. bench是google benchmark目录。

**Flag:** 在b站上开一门教学课程。