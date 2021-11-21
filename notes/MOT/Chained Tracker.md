# Chained Tracker

论文链接：[Chained-Tracker: Chaining Paired Attentive Regression Results for End-to-End Joint Multiple-Object Detection and Tracking](https://arxiv.org/abs/2007.14557)

### 一、简介

MOT检测器主要是由三个部分组成：目标检测、特征提取和数据联系，现存的检测要不就是三个分开进行要么就是融合两个，而原文提出的CTracker则是融合了三个部分为一个模型，同时CTracker还解决了目前MOT检测器越来越复杂的问题。

CTracker接收一对相邻的帧作为输入，同时对每一对对象进行边界框回归。为了进一步提高CTracker的性能，引入一个联合注意力模块来预测置信度映射。它使用两个分支模型来指导成对边界框回归分支关注空间区域信息。一个是目标分类分支，用于预测边界框对中的第一个框的是否为前景的置信度得分。另一个是ID验证分支，其评估边界框对中的目标是否为相同目标，从而促进回归分支的精度。最后边界框对根据分类分支置信度进行过滤，然后在两个相邻帧对中的公共帧中生成的边界框对使用IoU进行匹配关联。如此，追踪过程就可以通过链接所有的相邻帧对实现。

### 二、CTracker 实现

#### 2.1 Node Chaining

<img src="..\img\Node Chaining.jpg" style="zoom:70%;" />

和别的MOT相比，CTracker不是接收一帧而是相邻的两帧，值得注意的一点是最后依次输入属于的是最后一帧和最后一帧的复制。CTracker将相邻两帧中的bboxes进行匹配，因为是相邻帧，理论上来说两帧中同一个对象的bbox应该相差很小。匹配过程如下：第一帧的每一个bbox生成一个tracklet并随机赋予ID，之后就不断地成组进行匹配，匹配时获取IoU然后应用匈牙利算法。通过链接相邻节点获取目标轨迹，如果出现了没有匹配的点那么就创建新的tracklet。

##### Robustness增强

为了为了增强模型对各种遮挡（可能会导致中间帧的检测失败）和短期消失（后快速出现）鲁棒性，作者保留终止的轨迹和对应的ID σ 帧，在这σ 帧中持续使用一个简单的持续速度预测模型来进行运动估计，寻找他们的匹配目标。即一次匹配未找到匹配对象的tracklet将不会立刻去除，而是按照简单模型预测下一帧的轨迹然后加入到下一次匹配当中去，总共持续 σ 帧。

#### 2.2 网络架构

总体网络架构如下图所示：

<img src="..\img\CTracker.jpg" style="zoom: 80%;" />

##### Overview

如上图所示，首先使用了ResNet-50作为backbone来提取高级语义特征，然后集成FPN生成多尺度特征表征，用于后续的预测。为了相邻帧的目标关联，每个独立帧的不同尺度级特征图应该首先连接到一起，然后喂入预测网络对边界框对进行回归。见图3，成对边界框回归分支（上图右侧中间的分支）为每个目标返回一个边界框对， 目标分类分支（上图右侧上面的分支）预测一个目标是前景的置信度得分。目标分类和ID确认分支都是用来引导注意力的，从而避免无关混乱信息的干扰。

##### 成对box回归

目标检测中预测目标相对于anchor的偏移量，论文提出基于Chained-anchor的检测框分支，可以同时回归两个框。chained-anchor紧密的排列在空间网格上，每个chained-anchor都可以预测相邻帧中同一目标的两个检测框。为了解决真实场景中的尺度变换，作者用K-means算法聚类数据集中的标注框以得到chained-anchor的尺度。检测到的框首先要利用IOU通过soft-NMS过滤，再用分类得分过滤，最后输出的检测框对链接到整体轨迹。

##### 联合注意力模块

Joint Attention Module，为了在回归分支之前突出组合特征中的局部信息区域，作者引入了基于注意力机制的模块。如CTracker网络图所示，论文引入了id verification分支获取置信度得分，区分检测对中的两个框是否属于同一个对象。attention map由目标分类分支与id verification分支得到，这两个分支是互补的，目标分类分支的置信度图关注于前景区域即目标，id verification分支的预测则关注于同一目标的特征。

##### 特征重利用

由于CTracker的输入为相邻帧图片，为了避免双倍的计算与记忆损失，论文提出了Memory Sharing Mechanism (MSM)存储当前帧的特征并在下一个node中使用该特征。

最后关于CTracker的详细架构如下图所示：

<img src="..\img\Ctracker详细架构.jpg" style="zoom:80%;" />

#### 2.3 损失函数

分类损失函数使用focal loss，其分类损失label定义为如果大于一定阈值则为1，反之为0，而ID verification分支的损失函数也是用focal loss。而关于bbox回归的损失函数和Faster R-CNN一样，使用了smooth L1损失，其总损失函数如下：

<img src="E:\李云昊\国科\computer-vision\notes\img\CTracker 损失函数.jpg" style="zoom:80%;" />

α 和 β 都是可以设定的参数，在原文的实验中是取了1。







