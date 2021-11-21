# RetinaTrack: Online Single Stage Joint Detection and Tracking

论文链接：[RetinaTrack: Online Single Stage Joint Detection and Tracking](https://arxiv.org/abs/2003.13870)

### 一、简介

MOT任务现在大部分还是TBD范式，这些TBD生成的实时的MOT算法，往往在追踪时不去计算检测的时间，事实上，他们在检测所需的时间却很长。原文主要研究在自动驾驶方面的追踪，所以对速度和准确度要求都很高。RetinaTrack基于实时的RetinaNet检测器，加入实例级别嵌入进行数据关联。提出一个简单有效的post-FPN预测子网络来解决RetinaNet不适合对不同尺度的每个实例进行嵌入的问题。

### 二、RetinaTracker 架构

RetinaTracker 是一个RetinaNet的变体，使其可以提取实例级别的特征。和其他基于锚的检测相同，RetinaNet产生的检测也是锚框。为了将这些相邻帧的锚相关联，需要确定相应锚的特征向量，将他们传入特征嵌入网络中进行训练。其整体架构如下图所示：

![](..\img\RetinaTracker架构.jpg)

#### 2.1 RetinaNet

首先回顾一下RetinaNet，其具体介绍详见[Classic Algorithms](../Classic Algorithms.md)相关部分。在RetinaNet中我们使用了一个FPN-based特征提取器来生成多层特征图，然后每一层特征图都被送入两个卷积子网来各预测K个张量（一个对应一个anchor），然后就可以得到分类结果和bbox回归，其总体流程如上图a和b所示。值得注意的是别的论文大多将这些输出折叠为一个单一的组合张量，而不是k张量，每个锚形状一个张量。然而，为了清晰起见，原文将这些预测分开(最终结果是等效的)。

至此可以清晰地用公式表示RetinaNet从特征图（一层）上获取预测结果地过程：

![](..\img\RetinaTracker公式01.jpg)

#### 2.2 使用锚框级特征修改预测子任务

RetinaNet中K个锚框共享所有的卷积层参数来进行计算（直到最后的分类预测和bbox回归为止），因此，没有明确的方法来提取每个实例的特征，因为如果两个检测匹配相同位置的锚点具有不同的形状，那么在网络中唯一可以区分它们的点是在最后的类和框回归预测。原文的做法是在post-FPN预测之前强制的进行锚框分离（如上图c所示），使用中间级特征仍然能够可以唯一的关联一个锚框。作者相较于原本的网络结构做出了一点修改，采用了一点方式来进行限制权重。在RetinaTrack模型中，预测过程可以由以下公式表示：

![](..\img\RetinaTracker公式02.jpg)

也就是说，在RetinaTracker中对于每一个特征图层，我们首先并行应用K个m1层的卷积层来获取一个Tensor，这个张量可以视为per-anchor  instance-level  features，从这里开始就有一个唯一的张量关联每一个锚框。使用两个并行的task-specific post-FPN layers来处理每个Fi,k（就是刚刚提到的那个和锚框一一对应的张量），分别通过m2个3×3卷积和一个输出N分类通道或者一个4通道偏移回归的最终3×3卷积。

#### 2.3 嵌入结构

获取到了与锚框一一对应的张量之后作者又添加了第三个任务路线，由m3个1×1卷积层组成，用于将特征映射到最终的最终embedding空间，同样类似上述两节用公式表达流程如下：

![](..\img\RetinaTracker公式03.jpg)

同时作者在每一层（除了最后一层）后面都使用了batch norm和ReLU激活函数，同样的，这些卷积层参数对于不同FPN层级输出和K个anchors是共享的。这里的m3和之前的m1、m2都是超参数。

#### 2.4 训练

分类使用focal loss，边界框偏移回归使用Huber Loss（就是RetinaNet使用的损失函数）。对于embedding使用采样三组对比样本的批处理策略的三元组损失：

![](..\img\RetinaTracker公式04.jpg)

其中A是匹配上GT框的锚框数量，ty是匹配上的锚框y对应的追踪ID。Dab表示嵌入a和b之间的非平方的欧式距离。m是保护参数（m=0.1）。

#### 2.5 测试和跟踪逻辑

在测试时，首先创建一个track store用来保存轨迹信息，对于每个轨迹我们保存它的之前的检测结果（包括bbox、类别预测和得分）、embedding向量和track state。初始时候这个store是空的，然后在每一帧我们取RetinaTrack打分前100个检测的embedding向量，经过阈值筛选之后和track store中的track按照一定策略（比如实验中用的IoU大于0.4）进行匹配，进行track更新或者创建新的track。对于每个存活的track我们保存H个前检测三元组，后续的帧可以和H个检测结果中的任意一个匹配。Track最多存活40帧，如果40帧没有新的匹配出现则说明这个track死了。







