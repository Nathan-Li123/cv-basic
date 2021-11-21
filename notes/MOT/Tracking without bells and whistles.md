# Tracking without bells and whistles

论文链接：[Tracking without bells and whistles](https://arxiv.org/abs/1903.05625)

### 一、简介

作者提出将目标检测器转换为追踪器（只使用目标检测器进行跟踪），不特意去对当前存在的遮挡，重识别和运动预测进行优化而完成跟踪任务。且不需要进行训练和优化。**利用对象检测器的边界框回归来预测对象在下一帧中的新位置**，通过简单的重新识别和相机运动补偿对其进行扩展, 展示了跟踪器的可扩展性。作者提出的方法能够解决大部分简单的跟踪方案，并探究跟踪的一些挑战和问题。

### 二、Tracktor

作者的方法是将一个detector转化为tracktor，其核心就是一个regression-based检测器，原文中采用了backbone为ResNet-101的Faster R-CNN。在原本的Faster R-CNN中，特征图经过RoI之后会进入一个分类分支，分类分支会对proposals进行分类打分，而在tracktor中，这一分支被用来评估porposal是行人（对象）的可能性，回归分支还是和原本的一样进行bbox回归。最后还需要进行NMS来获取最终结果。

#### 2.1 流程

<img src="..\img\Trackor流程.jpg" style="zoom:80%;" />

在第一帧的时候并没有Tracker运行，为此Detection检测第一帧中存在的目标，并对目标进行初始化，记录入跟踪序列（bt）中。在第二帧到之后的所有帧中运行同样的步骤，作者先实现了对新检测的处理（上图中红色的部分），再处理跟踪部分，由于是一个循环的过程，所以顺序影响不是很大。

观察上图的蓝色线条，这展示的是处理跟踪的部分，第一步利用边框回归将t-1帧的活动轨迹扩展到第t帧。即通过将t-1的目标边界框回归到第t帧的新位置来实现。（作者假定目标在两帧间移动距离较短，利用检测网络能够捕捉到稍有移动的目标，对于Faster R-CNN来说就是在当前帧的特征图上使用上一帧的bbox坐标进行RoI操作）。标识将从以前的边界框自动转移到新回归的边界框, 从而有效地创建一个轨迹。第二步考虑哪些轨迹应当kill掉，这分为两种情况：一是新的分类score低于阈值时，二是新的和旧的bbox之间的IoU小于一定阈值时。

再看上图中红色线条，检测器对当前帧的所有目标进行检测获得Dt，如果新检测到的目标没有覆盖任何轨迹部分（之前帧的目标与当前帧其他目标的IOU小于阈值，也就是上一段说的轨迹结束的第二种情况），则认为该检测出来的目标为新目标，为其创建一个新的轨迹。

#### 2.2 tracking extentions

这是两个Tracktor非必要的组件，这两个组件能够提高Tracktor的精度，带有这两个组件的Trackor也被称为**Tracktor++**。

##### Motion model

之前的假设是在运动目标在帧运动距离较小，但是在相机大范围运动或者其他极端情况下会出现问题。作者所以提出能够加入两种模型来更好的估计目标在未来帧的位置。对于带有移动相机的序列,采用简单的相机运动补偿 (CMC)。

##### Re-ID

为了保持在线跟踪，使用了基于孪生网络的外观向量的短期重新身份识别算法（trinet）。在之前帧存储已经停用的轨迹，然后应用运动模型将新检测到的目标（轨迹）和停用的轨迹比较，通过计算空间距离（基于每个bbox的外观特征向量，使用Siamese CNN计算）来判断是要重启这个轨迹还是创建一个新的轨迹。

#### 2.3 Siamese 网络

这个Siamese网络并不是Trackor的必备组件，不过似乎很常用因此在这里记录一下。

在人脸识别中，存在所谓one-shot问题。举例来说，就是对公司员工进行人脸识别，每个员工只给你一张照片（训练集样本少），并且员工会离职、入职（每次变动都要重新训练模型）。有这样的问题存在，就没办法直接训练模型来解决这样的分类问题了。为了解决这种问题（不局限于人脸识别），可以**训练一个模型来输出两张图像的相似度**，而Siamese网络就是这种模型。

##### 原理

Siamese network就是连体的神经网络，所谓连体是通过共享权值来实现的，如下图所示：

<img src="..\img\连体神经网络.jpg" style="zoom:20%;" />

共享权值就是说上图中左右两个网络（都是由CNN或者LSTM构成的）的权重参数是完全一样的，在实现时甚至可以使用同一个网络。Siamese网络有两个输入，然后两个神经网络将输入映射到新的空间，形成在新的空间中的表示，通计算Loss（指的是两个网络输出的差异，比如向量欧氏距离）得到两幅图像的相似度。

##### 训练

Siamese网络的训练需要引入一个新损失函数。首先明确该网络的目标是使输入图像相似时输出距离较小，而输入图像不相似时输出距离较大。具体一点描述，我们选择一张图像作为anchor，然后找相似的（或者在人脸识别领域就是同一个人的）图像作为positive样本，不相似的作为negative样本，然后将损失函数定义为：

<img src="..\img\Siamese网络损失函数.jpg" style="zoom:80%;" />

其中α是正负样本之间的距离（超参数，他的意思应该是正负样本之间距离必须大于α），a、p、n分别代表anchor、positive和negative样本。损失函数包含了训练数据集中所有可能的三元组，有了这个就可以通过梯度下降进行训练了。值得注意的是这个训练数据集是不需要人工标注的，也就是说任何一张图像都可能是anchor、positive样本或是negative样本。




