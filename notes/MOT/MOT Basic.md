# MOT 基础

本文只是关于多目标追踪的一些基础知识记录（*参考网页写于2019年末，可能有所过时*），并作为后续学习的目录使用。

### 一、MOT入门

#### 1.1 简介

多目标跟踪，一般简称为MOT(Multiple Object Tracking)，也有一些文献称作MTT(Multiple Target Tracking)。在事先不知道目标数量的情况下，对视频中的**行人、汽车、动物等多个目标进行检测并赋予ID进行轨迹跟踪**。**不同的目标拥有不同的ID，以便实现后续的轨迹预测、精准查找等工作。**

#### 1.2 主要难点

目标跟踪是一个早已存在的方向，但之前的研究主要集中于单目标跟踪，直到近几年，多目标跟踪才得到研究者的密切关注。与其它计算机视觉任务相比，多目标跟踪任务主要存在以下研究难点：

* 数据集缺乏且标注困难
* 目标检测不够准确
* 频繁的目标遮挡
* 目标数量不确定
* 速度较慢，实时性不够；

#### 1.3 核心步骤

MOT算法的通常（应该说是经典的）工作流程：(1)给定视频的原始帧；(2)运行对象**检测器**以获得对象的边界框；(3)对于每个检测到的物体，**计算出不同的特征**，通常是视觉和运动特征；(4)之后，**相似度计算步骤**计算两个对象属于同一目标的概率；(5)最后，**关联步骤**为每个对象分配数字ID。

因此绝大多数MOT算法无外乎就这四个步骤：**①检测 ②特征提取、运动预测 ③相似度计算 ④数据关联**。其中影响最大的部分在于检测，检测结果的好坏对于最后指标的影响是最大的。

#### 1.4 评价指标

多目标追踪主要的评价指标如下图所示：

<img src="..\img\多目标追踪评估指标.jpg" style="zoom:40%;" />

第一个是传统的标准，现在已经没人用了，就不介绍了。

第二个是06年提出的CLEAR MOT。现在用的**最多**的就是**MOTA**。但是这个指标FN、FP的权重占比很大，更多衡量的是**检测的质量**，而不是**跟踪的效果**。

* MOTA：跟踪准确度，关注误报、错过目标和身份切换（公式中的IDSW就是身份错误数量）
* MOTP：跟踪精度，标注和预测bbox的不匹配度

第三个是16年提出的ID scores。因为都是基于匹配的指标，所以能更好的衡量**数据关联**的好坏。

* IDP、IDR：身份识别精度、身份回归率（公式表示地很清楚了）
* IDF1：见上图公式

#### 1.5 数据集

主要数据集包括MOTChallenage和KITTI，KITTI数据集是针对自动驾驶的数据集,有汽车也有行人，在MOT的论文里用的不多，而MOTChallenage数据集专注于行人追踪，MOT15都是采集的老的数据集的视频做的修正，而MOT16是全新的数据集，相比于15年的行人密度更高、难度更大，MOT17的视频和16年一模一样，只是提供了三个检测器，相对来说更公平。也是现在论文的**主流数据集**。19年的是针对特别拥挤情形的数据集，只有CVPR19比赛时才能提交。

#### 1.6 研究方案

视觉目标跟踪的发展相对较短，主要集中在近十余年。早期比较经典的方法有Meanshift和粒子滤波等方法，但整体精度较低，且主要为单目标跟踪。近五六年来，随着目标检测的性能得到了飞跃式进步，也诞生了**基于检测进行跟踪**的方案，并迅速成为当前多目标跟踪的主流框架，极大地推动了MOT任务的前进。同时，近期也出现了**基于检测和跟踪联合框架**以及**基于注意力机制的框架**，开始引起研究者们的注意力。

##### 基于Tracking-by-detection的MOT

基于Tracking-by-detaction框架的MOT算法是先对视频序列的每一帧进行目标检测，根据包围框对目标进行裁剪，得到图像中的所有目标。然后，转化为前后两帧之间的目标关联问题，通过IoU、外观等构建相似度矩阵，并通过匈牙利算法、贪婪算法等方法进行求解。代表方法是SORT和DeepSORT

##### 基于检测和跟踪联合的MOT

这种方式的典型实现是JDE，JDE采用FPN结构，分别从原图的 1/8，1/16 和 1/32 三个尺度进行预测。在这三个不同尺度的输出特征图上分别加入预测头(prediction head)，每个预测头由几层卷积层构成，并输出大小为 (6A+D)×H×W 的特征向量。其中 A 为对应尺度下设置的锚框的数量，D 是外观特征的维度。

##### 基于注意力机制的MOT

随着Transformer等注意力机制在计算机视觉中的应用火热，近期开始有研究者提出了基于注意力机制的多目标跟踪框架，目前主要有TransTrack和TrackFormer，这两项工作都是将Transformer应用到MOT中。TransTrack将当前帧的特征图作为Key，将前一帧的目标特征Query和一组从当前帧学习到的目标特征Query一起作为整个网络的输入Query。

### 二、论文阅读

##### 2019

ICCV2019：[Tracking without bells and whistles](./Tracking without bells and whistles.md)

##### 2020

ECCV2020：[Chained-Tracker: Chaining Paired Attentive Regression Results for End-to-End Joint Multiple-Object Detection and Tracking](./Chained Tracker.md)

ECCV2020：[Tracking Objects as Points](./CenterTrack.md)

CVPR2020：[RetinaTrack: Online Single Stage Joint Detection and Tracking](./Retina Tracker.md)

CVPR2020：[Learning a Neural Solver for Multiple Object Tracking](./Neural Solver for MOT.md)

##### 2021

IJCV2021：[FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking](./Fair MOT.md)

CVPR2021：[SiamMOT: Siamese Multi-Object Tracking](./SiamMOT.md)

