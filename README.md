# KuiperInfer
![](https://github.com/zjhellofss/kuiperinfer/actions/workflows/cmake.yml/badge.svg)


从零自制深度学习推理框架，学会深度学习框架背后的知识，学会怎么上手一个中等规模的C++项目。对于以后的面试和求职都是一个不错的项目哦！

项目整体风格和结构都具有C++项目的现代感，供大家学习参考。 视频课程链接：[https://space.bilibili.com/1822828582](https://space.bilibili.com/1822828582)

![](./imgs/logo.jpg)

## 项目贡献者

**如何参与项目的贡献？提交代码增加新功能或修改bug；提出有用的建议；完善文档或增加单元测试。**

- [zjhellofss](https://github.com/zjhellofss)

- [liuxubit](https://github.com/liuxubit)

- [Azusachan](https://github.com/Azusachan)
- [wfs2010](https://github.com/wfs2010)
- [mlmz](https://github.com/minleminzui)
## 课程计划

我在b站上开了一门教学课程，目前是课程的前八次课程，课程大纲如下，主页是: https://space.bilibili.com/1822828582 , 欢迎大家关注支持。

### 学习群：
请注明来意，我会拉你进群

![](https://i.imgur.com/7myOrcY.png)

| 课程节数 | 主要内容                   | 进度 | 课程链接                                         |
|------|------------------------|----|----------------------------------------------|
| 第一次课 | 整体框架解读和开发环境配置          | 完成 | https://www.bilibili.com/video/BV1HV4y1A7H8/ |
| 第二次课 | 张量Tensor类的解析和输入数据的内存排布 | 完成 | https://www.bilibili.com/video/BV1Ed4y1v7Gb/ |
| 第三次课 | 从CSV文件中初始化张量Tensor一个实例 | 完成 | https://www.bilibili.com/video/BV1Pg411J7V5/ |
| 第四次课 | 手写第一个算子Relu并完成算子注册工厂类  | 完成 | https://www.bilibili.com/video/BV1bG4y1J7sQ/ |
| 第五次课 | Im2col的原理和卷积算子的实现      | 完成 |      https://www.bilibili.com/video/BV1F841137Ct                                        |
| 第六次课 | 照猫画虎，完成MaxPooling算子    | 完成 |         https://www.bilibili.com/video/BV1m3411S7yy                                     |
| 第七次课 | 图结构(PNNX)讲解和计算图初步      | 完成 |   https://www.bilibili.com/video/BV1VW4y1V7vp                                           |
| 第八次课 | 读取PNNX并构建自己的计算图        | 完成 |          https://www.bilibili.com/video/BV1HY4y1Z7S3                                    |
| 第九次课 | 卷积算子的实现和im2col加速计算的原理       | 完成 |         https://www.bilibili.com/video/BV1F841137Ct                                    |
| 未完待续 | ......                 | ...... | ...... |

## 使用的技术和开发环境
* 开发语言：C++ 17
* 数学库：Armadillo + OpenBlas(或者更快的Intel MKL)
* 加速库：OpenMP
* 单元测试：Google Test
* 性能测试：Google Benchmark

## 安装过程(使用Docker)
1. docker pull registry.cn-hangzhou.aliyuncs.com/hellofss/kuiperinfer:latest
2. sudo docker run -t -i registry.cn-hangzhou.aliyuncs.com/hellofss/kuiperinfer:latest /bin/bash
3. cd code 
4. git clone  https://github.com/zjhellofss/KuiperInfer.git 
5. cd KuiperInfer
6. **git checkout -b 你的新分支 study_version_0.01 (如果想抄本项目的代码，请使用这一步切换到study tag)**
7. mkdir build 
8. cd build 
9. cmake -DCMAKE_BUILD_TYPE=Release -DDEVELOPMENT=OFF .. 
10. make -j16

**Tips:**

1. **如果需要对KuiperInfer进行开发**，请使用 git clone  --recursive https://github.com/zjhellofss/KuiperInfer.git 同时下载子文件夹tmp, 并在cmake文件中设置`$DEVELOPMENT`或者指定`-DDEVELOPMENT=ON`
2. **如果国内网速卡顿**，请使用 git clone https://gitee.com/fssssss/KuiperInferGitee.git 
3. **如果想获得更快地运行体验**，请在本机重新编译openblas或apt install intel-mkl

##  安装过程(不使用docker)
1. git clone --recursive https://github.com/zjhellofss/KuiperInfer.git
2. **git checkout -b 你的新分支 study_version_0.01 (如果想抄本项目的代码，请使用这一步切换到study tag)**
3. 安装必要环境(openblas推荐编译安装，可以获得更快的运行速度，或者使用apt install intel-mkl替代openblas)
```shell
 apt install cmake, openblas-devel, lapack-devel, arpack-devel, SuperLU-devel
```
4. 下载并编译armadillo https://arma.sourceforge.net/download.html
5. 编译安装glog\google test\google benchmark
6. 余下步骤和上述一致

**Tips:**
1. google benchmark编译过程中，如果遇到关于gtest缺失的报错，可以在google benchmark的cmake中关闭gtest选项

## 如何运行Yolov5的推理

请在demos文件夹下的yolo_test.cpp文件夹中以下代码进行修改

```cpp
const std::string& image_path = "imgs/car.jpg";
const std::string& param_path = "tmp/yolo/demo/yolov5s_batch8.pnnx.param";
const std::string& weight_path = "tmp/yolo/demo/yolov5s_batch8.pnnx.bin";
```

- `image_path`指定图像目录，`param_path`为模型的参数文件，`weight_path`为模型的权重文件，请替换为自己本地的路径。

- 模型定义和权重下载地址如下： https://cowtransfer.com/s/9bc43e0905cb40 

- 编译完成后，在项目目录调用 `./build/demos/yolo_test`

### 效果图
<img src="https://i.imgur.com/JkZ9KiE.jpg" alt="output0" style="zoom:50%;" />

## 已经支持的算子
**总体理念：逐步优化已经有的算子；有需要的时候再对未实现的算子进行开发**

- Convolution 
- AdaptivePooling 
- MaxPooling 
- Expression(抽象语法树)
- Flatten, View(维度展平和变形)
- Sigmoid 
- HardSigmoid 
- HardSwish 
- ReLU
- Linear(矩阵相乘)
- Softmax 
- BatchNorm
- Upsample
- SiLU
- Concat

## 目录
source是源码目录

1. **data/** 是张量类Tensor的实现和Tensor初始化方法
2. **layer/** 是算子的实现
3. **parser/** 是Pnnx表达式的解析类
4. **runtime/** 是计算图结构，解析和运行时相关

**test**是单元测试目录，基本做到public方法单元测试权覆盖

**bench**是google benchmark, 包含对MobilenetV3, Resnet18和yolov5s的性能测试。

## 性能测试
### 测试环境
#### 测试设备

15 核心的AMD EPYC 7543(霄龙) 32-Core Processor (Docker 容器，宿主机共有32核心)

#### 编译环境：

gcc (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0

### 性能结果

| **input size**         | **模型名称**     | **计算设备**              | **耗时**           |
|------------------------| ---------------- | ------------------------- |------------------|
| 224×224 batch = 8      | MobileNetV3Small | CPU(armadillo + openblas) | 7.77ms / image   |
| 224×224 batch = 8      | ResNet18         | CPU(armadillo + openblas) | 24.80ms / image  |
| 224×224 batch =16      | ResNet18         | CPU(armadillo + openblas) | 16.45ms / image  |
| 320×320 batch = 8      | Yolov5nano       | CPU(armadillo + openblas) | 13.39ms / image  |
| **640×640** batch = 8  | **Yolov5s**      | CPU(armadillo + openblas) | 209.93ms / image |
| **640×640** batch = 16 | **Yolov5s**      | CPU(armadillo + openblas) | 159.45ms / image |

## 致谢

推理框架NCNN，已经在借鉴的代码中保留了NCNN的BSD协议 https://github.com/Tencent/ncnn

优秀的数学库Openblas: https://github.com/xianyi/OpenBLAS

优秀的数学库Armadillo: https://arma.sourceforge.net/docs.html

给予我灵感的Caffe框架: https://github.com/BVLC/caffe