# KuiperInfer
![](https://github.com/zjhellofss/kuiperinfer/actions/workflows/cmake.yml/badge.svg)


从零自制深度学习推理框架，学会深度学习框架背后的知识，学会怎么上手一个中等规模的C++项目。对于以后的面试和求职都是一个不错的项目哦！

项目整体风格和结构借鉴了Caffe，仅用作学习参考。 视频课程链接：[https://space.bilibili.com/1822828582](https://space.bilibili.com/1822828582)

![](./imgs/logo.jpg)

## 项目贡献者
- [zjhellofss](https://github.com/zjhellofss)

- [liuxubit](https://github.com/liuxubit)

- [Azusachan](https://github.com/Azusachan)
- [wfs2010](https://github.com/wfs2010)
## 课程计划

我在b站上开了一门教学课程，目前是课程的第一季，课程大纲如下，主页是: https://space.bilibili.com/1822828582 , 欢迎大家关注。

| 课程节数    | 主要内容                   | 进度 | 课程链接                                         |
|---------|------------------------|----|----------------------------------------------|
| 第一次课    | 整体框架解读和开发环境配置          | 完成 | https://www.bilibili.com/video/BV1HV4y1A7H8/ |
| 第二次课    | 张量Tensor类的解析和输入数据的内存排布 | 完成 | https://www.bilibili.com/video/BV1Ed4y1v7Gb/ |
| 第三次课    | 从CSV文件中初始化张量Tensor一个实例 | 完成 | https://www.bilibili.com/video/BV1Pg411J7V5/ |
| 第四次课    | 手写第一个算子Relu并完成算子注册工厂类  | 完成 | https://www.bilibili.com/video/BV1bG4y1J7sQ/ |
| 第五次课    | Im2col的原理和卷积算子的实现      | 完成 |      https://www.bilibili.com/video/BV1F841137Ct                                        |
| 第六次课    | 照猫画虎，完成MaxPooling算子    | 完成 |         https://www.bilibili.com/video/BV1m3411S7yy                                     |
| 第七次课    | 图结构(PNNX)讲解和计算图初步      | 完成 |   https://www.bilibili.com/video/BV1VW4y1V7vp                                           |
| 第八次课    | 读取PNNX并构建自己的计算图        | 未完成 |          https://www.bilibili.com/video/BV1HY4y1Z7S3                                    |
| 第二季课程待叙 | ...                    | ... |                                              |

先列前八次(第一季)的课程，课程目录可能会发生变化。
后续课程会在第八次课程讲完后发布。

## 使用的技术和开发环境
* 开发语言：C++ 17
* 数学库：Armadillo+OpenBlas(或者更快的Intel MKL)
* 加速库：OpenMP
* 单元测试：GTest
* 性能测试：Google Benchmark

## 安装过程(已提供docker环境)
1. docker pull registry.cn-hangzhou.aliyuncs.com/hellofss/kuiperinfer:latest
2. sudo docker run -t -i registry.cn-hangzhou.aliyuncs.com/hellofss/kuiperinfer:latest /bin/bash
3. cd code
4. cd KuiperInfer
5. git pull (可选的命令)
5. mkdir build
6. cd build
6. cmake ..
7. make -j16
## 效果图
![](https://i.imgur.com/JkZ9KiE.jpg)

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
9. ReLU
10. Linear(矩阵相乘，只支持二维Tensor相乘，其他维度用到再开发)
11. Softmax 
12. BatchNorm
13. Upsample
14. SiLU
15. Concat

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
### 模型运行速度

|  Benchmark  | Batch Size | Time   | Image Size |
|  ----  |------------|--------|------------|
| BM_MobilenetV3_Batch8_224x224  | 8          | 0.081s | 224 × 224  |
| BM_Resnet18_Batch8_224x224  | 8          | 0.645s | 224 × 224  |
| BM_Resnet18_Batch16_224x224   | 16         | 1.569s | 224 × 224  |
| BM_Yolov5nano_Batch4_320x320  | 8          | 0.155s | 320 × 320  |
| BM_Yolov5s_Batch4_640x640   | 4          | 2.045s | 640× 640   |
| BM_Yolov5s_Batch8_640x640   | 8          | 4.740s | 640× 640   |
## 致谢
推理框架NCNN，已经在借鉴的代码中保留了NCNN的BSD协议 https://github.com/Tencent/ncnn

优秀的数学库Openblas: https://github.com/xianyi/OpenBLAS

优秀的数学库Armadillo: https://arma.sourceforge.net/docs.html

给予我灵感的Caffe框架: https://github.com/BVLC/caffe