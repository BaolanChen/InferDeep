<div align="center">
  
InferDeep
===========================
<h4>InferDeep：从零开始动手构建一个深度学习推理框架，支持Unet、Yolov5、Resnet等模型的推理。</h4>

<h4> Implement a high-performance deep learning inference library step by step，Make Our Hands Dirty！</h4>

---
<div align="left">

1. 学习一个深度学习框架背后的知识，掌握现代C++项目的写法，调试技巧和工程经验；
2. 如何设计、编写一个计算图；
3. 实现常见的算子，卷积算子、池化算子、全连接算子等；
4. 在3的基础上，学会常见的优化手段加速算子的执行；
5. 最后你将获得一个属于自己的推理框架，可以推理resnet, unet, yolov5, mobilenet等模型，对面试和知识进阶大有裨益。

**视频课程链接：**[https://space.bilibili.com/1822828582](https://space.bilibili.com/1822828582)


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
