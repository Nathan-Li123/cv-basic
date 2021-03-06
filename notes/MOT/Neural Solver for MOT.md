# Learning a Neural Solver for Multiple Object Tracking

论文链接：[Learning a Neural Solver for Multiple Object Tracking](https://arxiv.org/abs/1912.07515)

### 一、简介

这篇论文主要是将图论应用到MOT上去。目前MOT检测器的主流原则是tracking-by-detection，主要步骤为先对每一帧进行目标检测，然后链接他们形成轨迹，其中第二步就可以视为一个**图分割问题**：一个检测目标看作一个点，目标之间的链接视为边，有激活边连接的两个点是在一个轨迹上的，第二步就可以转化为找到图中所有连接的点集。解决图分割问题的方法也是两步的，首先为每一条边设置一个cost来衡量两点是一条轨迹上的可能性，然后寻找cost最小的分割集。以前基于图论的MOT的工作大致可分为两类：那些专注于图公式化的研究（更好的图优化框架），以及那些专注于学习更好成本的研究，而作者则是结合了两种方法，提出了在学习特征的同时，学习通过对整个图的推理来提供的解决方案。使用MOT的经典网络流公式来定义模型，直接预测图的最终分割成轨迹。过程中使用一个消息传递网络（MPN）来在图上组合深度特征为高阶信息，因此可以解释检测间的全局交互。

### 二、图问题解释追踪

原文的方法是建立在经典最小代价流上的的，因此首先介绍网络流MOT公式（network flow MOT formulation）。

#### 2.1 网络流公式

为不同时间戳上的每一对结点定义一个二分变量：

![](..\img\网络流公式二分变量.jpg)

这个公式的意义就是任意两点（这个点表示一个目标检测结果，包含对象、bbox坐标和时间戳，总共n个）的连线，如果其连接的两点属于相同轨迹并且在轨迹内暂时连续的设置为1，剩下的都设置为0。

#### 2.2 从学习代价到预测方案

标准的做法是为上一节所说的那个二分变量，也就是边赋予一个代价，代表是active的可能性。但是原文的目标是直接学习预测哪个边是active的，因此作者提出直接学习预测图中的哪条边被激活，即边上的二元变量的最终值，为此我们将这个任务视为边上的二分类任务，标签就是二元变量值。总之，我们利用经典网络流公式化将多目标跟踪视为一个完全可学习的任务。

### 三、实现

原文的主要贡献是构造了一个可区分的框架来像训练边分类器一样训练MOT，它主要有四个步骤：

* 图构建：给定一个视频中目标检测的集合，构建一个图，其中节点对应了检测，边对应检测之间的连接。
* 特征编码：在边界框图像上应用一个卷积神经网络，初始化节点的外观特征嵌入。对于每一条边也就是不同帧的每一对检测，我们计算一个具有编码了他们的边界框之间的相对大小，位置以及时间差的特征的向量。然后将其输入到一个多层感知器中得到边的几何嵌入。
* 神经信息传递：我们在整个图上执行了一系列的信息传递步骤。直觉上，对于每一轮的信息传递，节点会与他们的连接边分享外观信息，边会与他们的伴随节点分享几何信息。最后，能够获得节点和边的更新嵌入信息，其包含了依赖于整个图结构的高阶信息。
* 训练：利用模型对最后的边嵌入预测一个目标流变量的连续近似。然后，遵循一个简单的范式对他们进行四舍五入，获得最终的跟踪轨迹，并使用交叉熵损失进行训练。

<img src="E:\李云昊\国科\computer-vision\notes\img\流程01.jpg" style="zoom:80%;" />

#### 3.1 信息传递网络

图构建得到的每一个点都是一个目标检测结果对象，而每一个边都是一个链接，他们都有自己的嵌入信息，而MPN的目的就是将这些信息传播到整张图上。这个传播包含两阶段，点到边然后边到点，具体公式如下：

<img src="..\img\MPN公式.jpg" style="zoom:85%;" />

h代表的是node embedding而m代表的是edge embedding，而这个l指的是迭代次数，Ne和Nv这两个函数都是可学习的函数。经过L（原文经过实验最终取12）次迭代之后，每个结点都包含L距离内的结点的嵌入信息。

#### 3.2 时间感知信息传递

上一节表示的信息传递传递网络适用于随机网络，但是MOT图有一些特殊架构，比如说一个结点只能连接一个更早的节点和一个更晚的节点，因此需要一种新的信息传递方法，这就是时间感知信息传递，其总体流程如下：

![](E:\李云昊\国科\computer-vision\notes\img\时间感知信息传递.jpg)

将信息传递过程边临时嵌入信息m划分成两个部分，一个来自过去帧的节点，另一个来自未来帧的节点，因此新的m如下：

<img src="..\img\时间感知传递公式01.jpg" style="zoom:90%;" />

同样的，点的嵌入信息也是分成了两块，作者据这些节点是节点 i 的过去节点还是未来节点来分别合并这些临时嵌入：

![](..\img\时间感知信息传递公式02.jpg)

获取了过去特征嵌入和未来特征嵌入之后，将其连接并输入到最后一个多层感知机中，获得更新后的节点特征嵌入：

![](..\img\时间感知信息传递公式03.jpg)

#### 3.3 特征嵌入

还有很重要的一点首先要得到特征嵌入，分为外貌嵌入（节点嵌入）和几何嵌入（连接特征嵌入）：

* 外貌嵌入：使用CNN提取特征，对于每一个检测结果对象对应的图片补丁区域使用CNN计算嵌入。
* 集合嵌入：不同时间戳的两个检测对象，用左上角坐标和长宽参数化他们的边界框，计算边界框相对距离，然后将边界框相对距离、外貌相对距离、时间戳距离拼接起来使用CNN（和上面那个不一样）计算得到初始边嵌入。边界框相对距离计算如下：

<img src="..\img\边界框相对距离公式.jpg" style="zoom:90%;" />









