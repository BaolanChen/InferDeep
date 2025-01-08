<div align="center">
  
InferDeep
===========================
<h4>InferDeep：从零开始动手构建一个深度学习推理框架，支持Unet、Yolov5、Resnet等模型的推理。</h4>

<h4> Implement a high-performance deep learning inference library step by step，Make Our Hands Dirty！</h4>

---
<div align="left">

**视频课程链接：**

1. 学习一个深度学习框架背后的知识，掌握现代C++项目的写法，调试技巧和工程经验；
2. 如何设计、编写一个计算图；
3. 实现常见的算子，卷积算子、池化算子、全连接算子等；
4. 在3的基础上，学会常见的优化手段加速算子的执行；
5. 最后你将获得一个属于自己的推理框架，可以推理resnet, unet, yolov5, mobilenet等模型，对面试和知识进阶大有裨益。



## 课程大纲

第二次课程是第一次课程的重置版，内容更加充实和完善，第一次课程大纲见下方章节。

| 课程节数                                              | 进度  | 课程链接                                    |
| ----------------------------------------------------- |-----| ------------------------------------------- |
| **第一讲** 项目预览和环境配置                         | 完成  | https://www.bilibili.com/video/BV118411f7yM |
| **第二讲** 张量(Tensor)的设计与实现                   | 完成  | https://www.bilibili.com/video/BV1hN411k7q7 |
| **第三讲** 计算图的定义                               | 完成  | https://www.bilibili.com/video/BV1vc411M7Yp |
| **第四讲** 构建计算图关系和执行顺序                   | 完成  | https://www.bilibili.com/video/BV19s4y1r7az |
| **第五讲** KuiperInfer中的算⼦和注册⼯⼚                 | 完成  | https://www.bilibili.com/video/BV1gx4y1o7pj |
| **第六讲** 卷积和池化算子的实现                       | 完成  | https://www.bilibili.com/video/BV1hx4y197dS |
| **第七讲** 表达式层中词法分析和语法分析以及算子的实现 | 完成  | https://www.bilibili.com/video/BV1j8411o7ao |
| **第八讲** 自制推理框架支持Resnet网络的推理           | 完成  | https://www.bilibili.com/video/BV1o84y1o7ni |
| **第九讲** 自制推理框架支持YoloV5网络的推理           | 完成  |    https://www.bilibili.com/video/BV1Qk4y1A7XL                                        |

## Demo效果

### Unet语义分割

> 🥰 KuiperInfer当前已支持Unet网络的推理，采用[carvana的预训练权重](https://github.com/milesial/Pytorch-UNet)

![](https://imgur.com/FDXALEa.jpg)
![](https://imgur.com/hbbZeoT.jpg)

推理复现可参考文末的 **运行 Kuiper 的 demo**

### Yolov5目标检测

> Demo直接使用yolov5-s的预训练权重(coco数据集)，使用KuiperInfer推理



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
4. git clone --recursive https://github.com/zjhellofss/KuiperInfer.git
5. cd KuiperInfer
6. **git checkout -b 你的新分支 study_version_0.02 (如果想抄本项目的代码，请使用这一步切换到study tag)**
7. mkdir build
8. cd build
9. cmake -DCMAKE_BUILD_TYPE=Release -DDEVELOPMENT=OFF ..
10. make -j$(nproc)

**Tips:**

1. **如果需要对KuiperInfer进行开发**，请使用 git clone  --recursive https://github.com/zjhellofss/KuiperInfer.git 同时下载子文件夹tmp, 并在cmake文件中设置`$DEVELOPMENT`或者指定`-DDEVELOPMENT=ON`
2. **如果国内网速卡顿**，请使用 git clone https://gitee.com/fssssss/KuiperInferGitee.git
3. **如果想获得更快地运行体验**，请在本机重新编译openblas或apt install intel-mkl

## 安装过程(构建Docker镜像)
1. docker build -t kuiperinfer:latest .
2. docker run --name kuiperinfer -it kuiperinfer:latest /bin/bash
3. cd /app
4. 余下步骤参考上述安装过程的步骤4-10

##  安装过程(不使用docker)
1. git clone --recursive https://github.com/zjhellofss/KuiperInfer.git
2. **git checkout -b 你的新分支 study_version_0.01 (如果想抄本项目的代码，请使用这一步切换到study tag)**
3. 安装必要环境(openblas推荐编译安装，可以获得更快的运行速度，或者使用apt install intel-mkl替代openblas)
```shell
 apt install cmake, libopenblas-dev, liblapack-dev, libarpack-dev, libsuperlu-dev
```
4. 下载并编译armadillo https://arma.sourceforge.net/download.html
5. 编译安装glog\google test\google benchmark
6. 余下步骤和上述一致
