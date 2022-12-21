# KuiperInfer
自制的一款推理框架，项目整体风格和结构借鉴了Caffe，仅用作学习参考。


## 使用的技术和开发环境
开发语言：C++ 17

数学库：Armadillo+OpenBlas

加速库：OpenMP

单元测试：GTest

性能测试：Google Benchmark

## 安装过程
1. sudo apt install cmake libopenblas-dev liblapack-dev libarpack2-dev libsuperlu-dev

2. git clone --recursive https://github.com/zjhellofss/KuiperInfer.git

3. cd KuiperInfer

4. mkdir build

5. cmake ..

6. make -j8

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



## 课程计划
我在b站上开了一门教学课程，目前是课程的第一季，课程大纲如下，主页是: https://space.bilibili.com/1822828582 , 欢迎大家关注。

| 课程节数    | 主要内容                   | 进度  |课程链接|
|---------|------------------------|-----|---------------------|
| 第一次课    | 整体框架解读和开发环境配置          | 完成  | https://www.bilibili.com/video/BV1HV4y1A7H8/|
| 第二次课    | 张量Tensor类的解析和输入数据的内存排布 | 完成  | https://www.bilibili.com/video/BV1Ed4y1v7Gb/|
| 第三次课    | 从CSV文件中初始化张量Tensor一个实例 | 完成 | https://www.bilibili.com/video/BV1Pg411J7V5/?spm_id_from=333.999.0.0|
| 第四次课    | 手写第一个算子Relu并完成算子注册工厂类  | 未完成 | |
| 第五次课    | Im2col的原理和卷积算子的实现      | 未完成 | |
| 第六次课    | 照猫画虎，完成MaxPooling算子    | 未完成 | |
| 第七次课    | 图结构(PNNX)讲解和计算图初步      | 未完成 | |
| 第八次课    | 读取PNNX并构建自己的计算图        | 未完成 | |
| 第二季课程待叙 | ...                    | ... | |

先列前八次(第一季)的课程，课程目录可能会发生变化。
后续课程会在第八次课程讲完后发布。

## 性能
### 测试环境
Ubuntu 22.04, Intel(R) Core(TM) i7-11800H @ 2.30GHz, 32G Memory
### 输入大小
8x3x224x224, NCHW格式
### 模型运行速度


|  Benchmark  | Batch Size | Time                | CPU              |Iterations|
|  ----  |------------|---------------------|------------------|---------------|
| BM_Resnet18  | 8          | 1.4107 s ± 0.15s    | 0.7390 s ± 0.15s |5|
| BM_MobilenetV3 | 8          | 1.1920 s   ±  0.15s | 0.7269 s ± 0.15s         |5|


## 致谢
优秀的数学库Openblas: https://github.com/xianyi/OpenBLAS

优秀的数学库Armadillo: https://arma.sourceforge.net/docs.html

给予我灵感的Caffe框架: https://github.com/BVLC/caffe