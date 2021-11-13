# 经典算法

本文主要包括一些经典的算法、分类器、模型等等的论文笔记及概述。除本文所记录的之外还有[Faster R-CNN](./Faster R-CNN.md)、[FPN](./FPN for Object Detection.md)、[ResNet](./ResNet.md)、[VGG](./VGG.md)。

### 一、YOLO V3

论文地址：../resources/papers/YOLOv3 An Incremental Improvement.pdf 或 [论文链接](https://arxiv.org/abs/1804.02767)

#### 1.1 简介

##### bbox预测

和之前的 YOLO9000 一样，YOLOv3 用四个值表示bbox（中心点坐标和长宽），训练时使用**平方误差损失**，而对于 objectness 的预测YOLOv3 使用了**逻辑斯蒂回归**。策略是如果是bbox是和ground truth框重合最大的或者IoU超过了阈值（原文使用了0.5）那么预测为1。同时 YOLOv3 只为每一个gt box设置一个bbox。

##### 分类预测

关于分类预测有一点很不同的是YOLOv3不使用softmax（原文给出的理由是首先softmax不必要，其次是softmax的分类是每一个bbox一个类的，但是因为YOLOv3只为每一个gt box设置一个bbox，因此一个bbox可能包含多个类）而是使用了一个独立的logistic分类器，在训练时使用**二值交叉熵损失**。

##### 多尺度预测

YOLOv3使用三个尺度图像进行预测，它进行特征提取的思路和特征金字塔类似，在原有的特征提取器上增加了几层卷积网络，最后一层输出一个3d的tensor，包含bbox、objectness和分类的预测。接着，我们从前两个图层中得到特征图，并对它进行2次上采样。再从网络更早的图层中获得特征图，用element-wise把高低两种分辨率的特征图连接到一起。这样做能使我们找到早期特征映射中的上采样特征和细粒度特征，并获得更有意义的语义信息。

##### 特征提取器

YOLOv3的特征提取网络结合之前的Darknet-19和当时流行的残差网络ResNet，由3×3的1×1的卷积核组成，总共由53层，被称为**Darknet-53**。Darknet的效率和精确度都非常的高。

*YOLOv3是一个很不错的分类器，他不擅长得到完美得到结果，它在COCO数据集上的 mAP[.5, .95]的表现并不是很好，但是他在mAP[.5]上的表现很好，可以和RetinaNet媲美，同时速度还很快。*

#### 1.2 YOLOvs3架构

<img src=".\img\YOLOv3架构.jpg"  />

其中DBL指的是Darknetconv2d_BN_Leaky，是yolo_v3的基本组件，就是卷积+BN+Leaky relu。

可以看到YOLOv3总共输出3个特征图，第一个特征图下采样32倍，第二个特征图下采样16倍，第三个下采样8倍。输入图像经过Darknet-53（无全连接层），再经过Yoloblock生成的特征图被当作两用，第一用为经过3*3卷积层、1*1卷积之后生成特征图一，第二用为经过1*1卷积层加上采样层，与Darnet-53网络的中间层输出结果进行拼接，产生特征图二。同样的循环之后产生特征图三。

#### 1.3 代码

GitHub项目[PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)，环境配置直接跟着项目中的README文档走即可，需要使用poetry包管理工具。

### 二、SSD

论文地址：../reousrces/papers/SSD Single Shot MultiBox Detector.pdf 或 [论文链接](https://arxiv.org/abs/1512.02325)

#### 2.1 简介

目前检测器的主要步骤一般为首先假设bbox，然后进行重新取样或回归，最后应用一个好的分类器，这种方式非常精确但是计算复杂度很高，即便硬件条件很好检测速度（通常用seconds per frame，SPF来衡量）也很慢，不适合实时应用。

而原文提出的SSD就是第一个不依赖上述步骤（从而加速）但是能达到相近精度的检测器。SSD的改进包括使用一个小的卷积滤波器来预测物体类别和边界框位置的偏移量，使用独立的预测器(过滤器)用于不同的宽高比检测，并将这些过滤器应用到网络的后期阶段的多个特征图，以便在多个尺度上执行检测。

#### 2.2 SSD 模型架构

<img src=".\img\SSDjiagou.jpg" style="zoom:80%;" />

SSD是基于一个产生固定数量的bbox以及其中物体的分类分数的前馈卷积网络，之后再通过一个 NMS 来产生最终结果。这些被称为基本网络，而在此之上还需要添加一些关键组件：

##### 多尺度特征图

在基本网络后加一组特征卷积层，这些卷积层逐渐减小size使得SSD能够在不同尺度作出预测。应该就是类似于FPN的思想，不同尺度的特征图上都做出预测。

##### 用于检测的卷积预测器

如上文所述，在新添加的每个不同尺度的卷积层上都滑动一个3×3的卷积核（padding为，也就是输入输出尺寸不变）来输出目标分类分数（比如共有c类目标，会得出c+1个score，SSD是要计算背景的得分）和bbox相对先验框（关于默认框详见下文）坐标偏移量。

##### 先验框和尺寸比例

SSD借鉴了Faster R-CNN中anchor的理念，但是SSD的先验框是应用在多个尺度特征图上的，每个单元设置尺度或者长宽比不同的先验框，预测的边界框（bounding boxes）是以这些先验框为基准的，在一定程度上减少训练难度。一般情况下，每个单元会设置多个先验框，其尺度和长宽比存在差异。

#### 2.3 训练

SSD和其他使用 region proposal 的检测器的训练的最关键的区别是SSD的**ground truth信息必须被分配到探测器的固定输出中的指定数据集**，一旦这种分配固定下来，接下来就是端对端训练了。除此之外还要考虑的是先验框的数量和尺寸以及难例挖掘和数据增强策略。

##### 匹配策略

训练过程中需要将default box和ground truth box相对应起来，对于每一个gt框都需要选择一些不同尺度、比例和位置的先验框。SSD的匹配规则和RPN类似，分为两步：首先对于每个ground truth，找到一个与其IoU最大的default box匹配，这个default box对应的预测框就作为正样本，其余都作为负样本。但这样做会导致负样本过多，正负样本分布极其不均衡，所以还会采取第二步：对于每个ground truth，将与其IoU大于某一阈值（比如0.5）的default box都进行匹配。

##### 损失函数

损失函数分为置信度损失（即分类损失，conf）和定位损失（loc）：

<img src="https://pic4.zhimg.com/80/v2-88e7c9a4b51f15d159a31ca7bc788497_720w.png" alt="img" style="zoom: 33%;" />

N为匹配成功的default box个数； α 是为了权衡两者损失而设置，通过交叉验证发现设置为1更好；第一项是置信度的Softmax损失，注意还包括背景这个类别；第二项是参数化后的bounding box中心坐标、长和宽的Smooth L1损失。

##### 选择default box尺度和比例

SSD在每个feature map上都可以选择不同的default box，下面假设共有 ![[公式]](https://www.zhihu.com/equation?tex=m) 个feature map，当前feature map序号记为 ![[公式]](https://www.zhihu.com/equation?tex=k) ，default box**相对于原图的比例**记为 ![[公式]](https://www.zhihu.com/equation?tex=s_k) ，则：

<img src="https://pic3.zhimg.com/80/v2-a76fdce058c83b393e958726b3e7ceb6_720w.png" alt="img" style="zoom:33%;" />

其中 ![[公式]](https://www.zhihu.com/equation?tex=s_%7Bmin%7D) 为最小尺度0.2， ![[公式]](https://www.zhihu.com/equation?tex=s_%7Bmax%7D) 为最大尺度0.9。长宽比设为5个值，于是default box的宽和高可以通过缩放比例和长宽比计算，除此之外，还增加一个长宽比为1，尺度为 ![[公式]](https://www.zhihu.com/equation?tex=s_k%27%3D%5Csqrt%7Bs_ks_%7Bk%2B1%7D%7D) 的default box，所以feature map上每个像素都有6个default boxes。上述数值也可以根据特定数据集进行设置。

##### 难例挖掘

将负样本根据confidence进行排序，从高到低选取负样本使正负样本比例至少为1:3，只有被选取的样本才能计算损失函数参与训练。

##### 数据增强

为了**获取小目标训练样本**，每张训练图片都会随机选择下面三个选项中的一个：

1. 使用整张图片
2. 采样一个patch，使其与物体的最小IoU为0.1、0.3、0.5、0.7或0.9
3. 随机选取一个patch

在得到patch之后再resize到固定尺寸，并以0.5的概率将其水平翻转。

### 三、Mask R-CNN

论文地址：../resources/papers/Mask R-CNN.pdf 或 [论文链接](https://arxiv.org/abs/1703.06870)

#### 3.1 简介

实例分割需要做到较好的完成检测任务的同时，并能够很好的分割实例（关于实例分割详见[Notes：实例分割](#####实例分割)），是物体检测和语义分割的集成。

Mask R-CNN在Faster R-CNN的基础上，对每个RoI增加一个掩码预测分支。这个分支用的是简单的FCN，可以实现像素到像素产生分割掩码，可是要在ROI区域进行一个mask分割，存在一个问题，Faster R-CNN不是为网络输入和输出之间的像素到像素对齐而设计的，如果直接拿Faster R-CNN得到的ROI进行mask分割，那么像素到像素的分割可能不精确，因为应用到目标检测上的核心操作执行的是粗略的空间量化特征提取，直接分割出来的mask存在错位的情况，所以作者提出了简单的，量化无关的层，称为RoIAlign(ROI对齐)，可以保留精确的空间位置，可以将掩码(mask)准确度提高10％至50％。同时通过这种方法Mask R-CNN将添加mask和类预测分开，这样效率会更高。

#### 3.2 Mask R-CNN 架构

Mask R-CNN 对于 Faster R-CNN 的 RPN 部分没有改动，它修改的是 RoI 部分，同时进行分类预测、bbox偏移和掩码生成。相应的，训练时的损失函数就变成了三个损失的和。掩码生成的分支输出一个k×m×m的张量（m×m分辨率，k个分类），然后在每个像素上进行一个sigmoid激活，最后的损失计算使用平均二值交叉熵损失。

##### RoI Align

RoI Pool是从每个ROI中提取特征图（例如7*7）的标准操作，为此需要做两次量化操作：1）图像坐标 — feature map坐标，2）feature map坐标 — ROI feature坐标。量化操作可以理解为取整操作，比如说图像坐标转化为feature map坐标时，以VGG16做backbone为例，经过4个池化层因此要除以16，虽然整个图像的尺寸一定能除尽，但是特征图像边框除以16不一定能除尽，这样就会产生浮点数而需要取整，一取整就产生了第一步量化误差（虽然看起来只差了零点几的误差，但是放大到正常图像上就比较大了）。第二步的量化误差也类似，这些误差对于掩码生成这种per-pixel的操作来说是不可接受的，因此Mask R-CNN使用RoIAlign。

RoI Align的思路是不使用量化操作，得到了一个浮点数那就使用这个浮点数，不进行取整操作。至于如何使用浮点数，Mask R-CNN 的解决思路是使用**“双线性值”算法**。双线性插值是一种比较好的图像缩放算法，它充分的利用了原图中虚拟点四周的四个真实存在的像素值来共同决定目标图中的一个像素值。

<img src=".\img\RoIAlign.jpg" style="zoom: 40%;" />

如上图所示，蓝色的虚线框表示卷积后获得的feature map，黑色实线框表示ROI feature(也就是每次pool覆盖的位置)，最后需要输出的大小是2x2，那么我们就利用双线性插值来估计这些蓝点（虚拟坐标点，又称双线性插值的网格点）处所对应的像素值，最后得到相应的输出。这些蓝点是2x2Cell中的随机采样的普通点，然后在每一个橘红色的区域里面进行max pooling或者average pooling操作（用哪个对结果影响不大），获得最终2x2的输出结果。

##### 网络架构

Mask R-CNN 的网络架构第一部分使backbone，主要使用的是和 Faster R-CNN 一样的架构，使用4-step training 的 ResNet-50，同时作者还探讨了使用当时新提出的特征金字塔 FPN。

第二部分是网络head部分，也就是上一节探讨的添加一个生成掩码的分支。具体如下图所示：

<img src=".\img\掩码生成架构.jpg" style="zoom:70%;" />

对于左边的架构，backbone使用的是预训练好的ResNet，使用了ResNet倒数第4层的网络。输入的ROI首先获得7x7x1024的ROI feature，然后将其升维到2048个通道（这里修改了原始的ResNet网络架构），然后有两个分支，上面的分支负责分类和回归，下面的分支负责生成对应的mask。由于前面进行了多次卷积和池化，减小了对应的分辨率，**mask分支开始利用反卷积进行分辨率的提升**，同时减少通道的个数，变为14x14x256，最后输出了14x14x80的mask模板。对于右边的架构，它的backbone使用的是FPN网络，mask分支中进行了多次卷积操作，首先将ROI变化为14x14x256的feature，然后进行了5次相同的操作，然后进行反卷积操作，最后输出28x28x80的mask。即输出了更大的mask，与前者相比可以获得更细致的mask。

*掩码分支可以预测每个RoI的K个掩码，但是我们只使用第k个掩码，其中k是分类分支预测的类别。*

#### 3.3 应用

Mask R-CNN 框架很容易被扩展应用到**人体姿态估计**中去，通过将计算关键点位置视为通过 Mask R-CNN 生成one-hot掩码（之前提到的是二值掩码）。应用时 Mask R-CNN 几乎不需要改动，只是让它为K个keypoints生成k个mask，然后拼接就完成了。但要注意的是在进行语义分割时这k个keypoints是独立进行的。

### 四、RFCN

论文地址：../resources/papers/R-FCN Object Detection via Region-based Fully Convolutional Networks.pdf 或 [论文链接](https://arxiv.org/abs/1605.06409)

*ps：这篇论文的模型介绍部分真心不好理解，因此参考了文章[详解 R-FCN](https://zhuanlan.zhihu.com/p/30867916)*

#### 4.1 简介

目标检测的网络可以被RoI pooling划分为两部分，一是独立于RoIs的共享的全卷积子网络，二是不共享计算的RoI子网络。之前的Faster RCNN对Fast RCNN产生region porposal的问题给出了解决方案，并且在RPN和Fast RCNN网络中实现了卷积层共享。但是这种共享仅仅停留在第一卷积部分，RoIpooling及之后的部分没有实现完全共享，可以当做是一种“部分共享”，这导致两个损失：1.信息损失，精度下降。2.由于后续网络部分不共享，导致重复计算全连接层等参数，时间代价过高。

Region-based Fully Convolutional Network (R-FCN) 试图以Faster RCNN和FCN为基础进行改进。

#### 4.2 R-FCN 架构

##### 总体框架

<img src=".\img\RFCN架构.jpg" style="zoom:67%;" />

如上图所示，原始图片经过conv卷积得到feature map1，其中一个subnetwork如同FastRCNN：使用RPN在featuremap1上滑动产生region proposal；另一个subnetwork则继续卷积，为每一类得到k×k深度的featuremap2，根据RPN产生的RoI(region proposal)在这些featuremap2上进行池化和打分分类操作，得到最终的检测结果。

##### 位置敏感得分图

这是R-FCN的主要思想技术。如果一个RoI含有一个类别c的物体，那么作者将该RoI划分为 k×k 个区域，分别表示该物体的各个部位，比如假设该RoI中含有人这个物体，k=3，那么就将“人”划分为了9个子区域，top-center区域毫无疑问应该是人的头部，而bottom-center应该是人的脚部，而将RoI划分为 k×k 个区域是希望这个RoI在其中的每一个区域都应该含有该类别c的物体的各个部位，即如果是人，那么RoI的top-center区域就必须含有人的头部。而当这**所有子区域都含有各自对应的该物体的相应部位**后，那么分类器才会将该RoI判断为该类别。具体流程如下图所示：

<img src=".\img\位置敏感得分图.jpg" style="zoom: 67%;" />

R-FCN会在共享卷积层的最后再接一层卷积层，它的height和width和共享卷积层的一样，但是它的输出channels= k×k×(c + 1)，表示c类再加上背景类，每个类都有k×k个score maps。我们假设这个类是人，每一个score map表示“原图image中的哪些位置含有人的某个部位”。那么我们只要将RoI的各个子区域对应到“属于人的每一个score map”上然后获取它的响应值就好了

##### 位置敏感池化

通过RPN提取出来的RoI区域包含中心坐标、长宽四个属性，依此定位到特征图上去。在R-FCN中，一个RoI会被分成k×k个bins（子区域），每个bin都会被对应到score map上的某一区域（因为在这里的特征图中，不同子区域对应的score map区域是在不同的层上面的），然后在score map上进行池化操作，这个池化操作是在**bin范围内进行平均池化**（也可以用max pooling），该操作对每个类都要进行，池化操作结束之后就得到一个channels为c+1，尺寸为k×k。也就是说对于每个类别有k×k个值，这些值相加就得到该类别的分数（这步就是图中的vote），最后应用一个softmax函数得到结果。

##### regression

除了上述的哪个位置敏感特征图的卷积层之外，还添加了一个sibling的4×k×k的卷积层用于bbox回归。它的思路和分类的思路是一样的，我们把4个属性看作4个类，同样地经过“position-sensitive score map”+“Position-sensitive RoI pooling”之后能得到四个类的评分，也就是"该RoI的坐标和长宽的偏移量"。

##### À trousand stride

作者将最后1个池化层的步长从2减小到1，那么图像将从缩小32倍变成只缩小16倍，这样就提高了共享卷积层的输出分辨率，而这样做就要使用Atrous Convolution算法。

#### 4.3 训练和测试

##### 训练

R-FCN的训练是简单的端对端训练。每个RoI损失是分类的交叉熵损失和bbox回归损失的加权和（背景不计入bbox回归损失）。IoU大于等于0.5的是正样本，反之则是负样本。同时，训练中很容易使用 online hard example mining（详见Notes：[OHEM](#####OHEM)），这是因为R-FCN中每一个RoI的计算量几乎可以忽略不计，因此训练时间不怎么会受影响。训练时使用0.0005的decay和0.9的冲量，默认情况下输入图像被resize为最短边为600。fine-tune时，使用0.001的学习率训练20k张，再用0.0001学习率训练10k张。而为了R-FCN和RPN共享特征，R-FCN采用了和 Faster R-CNN 一样的四步训练法。

##### 测试

在测试的时候，为了减少RoIs的数量，作者在RPN提取阶段就将RPN提取的大约2W个proposals进行过滤，一般只剩下300个RoIs，当然这个数量是一个超参数。并且在R-FCN的输出300个预测框之后，仍然要对其使用NMS去除冗余的预测框。

### 五、Cascade R-CNN

论文地址：../resources/papers/Cascade R-CNN Delving into High Quality Object Detection.pdf 或 [论文链接](https://arxiv.org/abs/1712.00726)

#### 5.1 简介

在之前的目标检测器中，随着IoU阈值的提升检测器表现会变差，这主要有两个原因：一是因为阈值提高而出现指数级别增长的消失正样本而导致的over fitting，二是一个mismatch问题（raining阶段和inference阶段，bbox回归器的输入分布是不一样的，training阶段的输入proposals质量更高(被采样过，IoU>threshold)，inference阶段的输入proposals质量相对较差。这是一个固有问题，但是阈值高的时候比较严重）。

Cascade R-CNN就是为了解决这些问题而提出的，它包含**一些使用递增的IoU阈值训练的检测器**（每一级的detector对于检测其响应的IoU的bbox效果最好），训练和测试的时候也是一级级进行的。训练时使用前一级的输出来测试下一级，这样能有效解决正负样本不均衡的问题，而测试时也是用同样的方法，这也能很好的解决mismatch问题（mismatch问题在IoU为0.5或者更低时不明显）。

<img src="E:\李云昊\国科\computer-vision\notes\img\Cascade R-CNN.jpg" style="zoom:80%;" />

如上图所示，有关于bbox回归Cascade R-CNN 的做法是为每一级分别设置一个specialized regressors，注意每一级的回归计算都是基于上一级的输出的而不是初始输入。

*Cascade R-CNN并无什么架构可言，它的主要思想就是以上的级联思想，其检测器理论上来说可以是任何一个基于R-CNN的检测器*

### 六、Retina Net

论文地址：../resources/papers/Focal Loss for Dense Object Detection.pdf 或 [论文链接](https://arxiv.org/abs/1708.02002)

#### 6.1 简介

Retina Net 是第一个达到甚至超过最新的two-stage探测器的one-stage探测器，事实上Retina Net 的主要贡献是设计了一个新的Loss函数。目前 one-stage 的精度不够的主要原因是正负样本不均衡，two-stage 的探测器在生成框阶段使用Selective Search, EdgeBoxes, RPN的结构极大的减少了背景框的数量，使其大约为1k~2k。在分类阶段，使用一些策略，如使前景背景的比例为1:3或者OHEM算法，这样就使得正负样本达到了一个平衡。但是One-Stage算法在进行将采样的同时产生预选框，在实际中经常会产生更多的框，但是真正的正样本的框却很少，就造成了样本间的极度不平衡，虽然有时会使用bootstrapping和hard example mining，但是效率很低。

原文提出了一种新的损失函数 Focal Loss，这是一个**动态变化尺度的交叉熵损失**，值得注意的是它的确切形式并不重要，可以有不同的是实现方式。除此之外，Retina Net 还使用了anchors和FPN的概念。

#### 6.2 Focal Loss

当样本不均衡的时候，如负样本很大，而且很多都是容易分类的（置信度很高的）,这样模型的优化方向就不是我们想要的方向，我是想让正负样本分开的，所以我们要把很多的注意力放在困难、难分类的样本上，所以作者在标准交叉熵损失的基础上进行了改进，首先我们把交叉熵二分类loss定义为：

<img src="https://pic1.zhimg.com/80/v2-56d5a848cb8694a897ef0688c916293c_720w.jpg" alt="img" style="zoom:80%;" />

然后 ![[公式]](https://www.zhihu.com/equation?tex=y%5Cepsilon) {-1, 1}表示正负样本的标签， ![[公式]](https://www.zhihu.com/equation?tex=p) 表示模型预测 ![[公式]](https://www.zhihu.com/equation?tex=y%3D1) 的概率，所以我们定义 ![[公式]](https://www.zhihu.com/equation?tex=p_t) 如下：

<img src="https://pic1.zhimg.com/80/v2-801ac87e391865452c9f5a98f3d1b990_720w.jpg" alt="img" style="zoom:80%;" />

然后我们就可以重写交叉熵损失为 <img src="https://www.zhihu.com/equation?tex=CE%28p%2Cy%29%3DCE%28p_t%29%3D-log%28p_t%29" alt="[公式]" style="zoom:80%;" />。

##### 平衡交叉熵损失

首先解决正负样本均衡的问题，我们为交叉熵损失函数添加一个权重参数 α，也就是说y为1时loss乘α，而y为-1时乘1-α。

##### Focal Loss 定义

添加权重参数的方法虽然平衡了正负样本，但是没有解决难易样本区分的需求。我们希望那些容易分的样本（置信度高的）提供的loss小一些，而那些难分的样本提供的loss几乎不变化，让分类器优化的方向更关注那些难分的样本。因此定义Focal Loss如下：

<img src="https://pic3.zhimg.com/80/v2-22d6c936444ec4bc917b38e64f671d26_720w.jpg" alt="img" style="zoom:80%;" />

当pt接近1，也就是预测正样本为正样本的概率接近1，预测负样本为正样本的概率接近0时（易分类样本），计算结果接近0，也就是使易分类样本的权重降低。而聚焦参数 γ 是用来平滑简单样本权重下降的速率（当其为0时就是交叉熵损失，实验表明 γ 为2时效果最好）。之后我们在为其加上权重参数：

<img src="https://pic1.zhimg.com/80/v2-3a360c548dd94cada980a25cd3062038_720w.jpg" alt="img" style="zoom:80%;" />

这就是原文使用的Focal Loss，不过正如前文所说，Focal Loss 的具体表达式不是唯一的。还可以有其他实现形式。

#### 6.3 Retina Net 架构

<img src="E:\李云昊\国科\computer-vision\notes\img\Retina Net 架构.jpg" style="zoom:80%;" />

Retina Net 本质上是由一个backbone网络和两个FCN子网络构成，backbone网络是现成的，负责卷积计算一个特征图，第一个子网基于特征图卷积计算目标分类，第二个则是作bbox回归。

##### FPN Backbone

在backbone部分可以使用特征金字塔来进行特征提取，典型的是在ResNet架构上搭建FPN。

##### Anchors

同时还可以在FPN的每一级上面使用anchors，这个和RPN的anchor是类似的，每一个anchor分配一个长度为K（K是对象分类数）的vector作为分类信息，以及一个长度为4的bbox回归信息。生成这些信息的方式和RPN只是修改为多尺度的并修改了IoU阈值。

##### 分类子网

分类子网对每个anchor的每个分类进行概率估计，其本质上是一个attach在每一个FPN层级上的小型FCN，其参数在每一层上是共享的。分类子网接受channels为C的特征图（滑动窗口的一个空间位置上 Anchor数为A，分类数为K），使用四个3×3的卷积层（channels为C），每层使用一个ReLU激活函数，然后再经过一个3×3的卷积层得到A×K个通道的分类结果。

##### bbox 回归子网

bbox回归子网和分类子网的架构基本相同，只不过最后输出的是4×A通道的结果。还有一点是Retina Net在bbox回归时时class-agnostic的，这样能够减少参数。

### 七、Center Net

论文地址：../reources/papers/Objects as Points.pdf 或 [论文链接](https://arxiv.org/abs/1904.07850)

参考链接：[扔掉anchor！真正的CenterNet——Objects as Points论文解读](https://zhuanlan.zhihu.com/p/66048276)

#### 7.1 简介

目前检测器检测对象的方法是构造一个紧贴着目标对象、坐标轴对齐的bbox，然后对框内对象进行分类识别。而原文提出了一种新的简单的检测器，它用bbox的中心点来表示一个物体，然后在中心点位置回归出目标的一些属性，例如：size, dimension, 3D extent, orientation, pose。 而**目标检测问题变成了一个标准的关键点估计问题**。我们仅仅将图像传入全卷积网络，得到一个热力图，热力图峰值点即中心点，每个特征图的峰值点位置预测了目标的宽高信息。

除此之外原文还对模型做了一些拓展：对于3D BBox检测，我们直接回归得到目标的深度信息，3D框的尺寸，目标朝向；对于人姿态估计，我们将关节点（2D joint）位置作为中心点的偏移量，直接在中心点位置回归出这些偏移量的值。

Center Net 的那个center point比较类似anchor，但是有几点不同：一是center point只基于位置，和box overlap无关；二是对于每一个位置都只取一个点；三是Center Net输出的heatmap分别率较高（输出步长为4，而一般的检测器输出步长为16）。

***Center Net 主要思想不是一个检测器，而是一个检测模型或者说一个检测思路（一种表示对象的方式），它在one-stage和two-stage的检测器上都可以应用。***

#### 7.2 CenterNet 实现

##### 关键点估计

首先设 I 为输入图像（宽为W，高为H，通道数为3），我们的目标是生成**热力图 Y** ，热力图尺寸为 W/R × H/R × C，其中R是输出步长（也就是上文提到的通常取4），C是关键点类别数（就是目标类别数）。热力图取值范围为0或1，1代表是当前坐标检测到了这种类别的物体，0则代表不存在。原文使用三种全卷积网络实现热力图预测：Resnet-18 with up-convolutional layers、DLA-34 、Hourglass-104。对于 每个Ground Truth某一类的关键点 c ,其位置为p ，计算得到低分辨率（经过下采样)上对应的关键点，我们将 GT 关键点通过高斯核分散到热力图上。如果有两个关键点重合（同类别），我们取元素级较高的那个（应该就是取高斯核计算出来更大的那个）。有关于高斯分布到热力图上这么说可能不是很好理解，那么直接看一个官方源码中生成的一个高斯分布[9,9]：

<img src="E:\李云昊\国科\computer-vision\notes\img\高斯分布.jpg" style="zoom:67%;" />

##### 损失函数

###### 中心点预测损失

训练时关于中心点预测使用像素级的focal loss：

![](.\img\Center Point 损失函数.jpg)

其中α和β是focal loss的超参，N是图像中的关键点数量。关于是中心点的损失函数很好理解，而不是中心点就稍显复杂。观察在otherwise条件下的表达式， 预测值越大（预测错了他还很肯定，说明是难例）则损失权重也越大，而另一项 <img src="https://www.zhihu.com/equation?tex=%281-%7BY%7D_%7Bx+y+c%7D%29%5E%7B%5Cbeta%7D" alt="[公式]" style="zoom: 67%;" />则是对中心点周围点做出了调整，因为靠得越近的点越容易干扰到实际中心点。同时这个也起到了处理正负样本不均衡的作用，在这里每一个物体只有一个实际中心点，其余的都是负样本，但是负样本相较于一个中心点显得有很多。

###### 目标中心偏移损失

因为上文中对图像进行了的下采样，这样的特征图重新映射到原始图像上的时候会带来精度误差，因此原文还为每一个center point添加了一个偏移量 O（所有类的偏移量相同)，注意 O 和热力图同样长宽但是通道数为2。偏移量训练使用L1损失，它只在关键点位置上作监督操作，其他位置无视。

###### 目标大小预测损失

此外还需要为每个目标k回归出目标的尺寸，为了减少计算负担，原文对每个目标种类使用单一的尺寸预测 S，S的尺寸和之前提到的偏移量 O 相同。关于S的回归原文也使用了L1损失。

###### 总体损失

我们不将scale进行归一化，直接使用原始像素坐标。为了调节该loss的影响，将其乘了个系数，整个训练的目标loss函数为：

![](.\img\Center Net 完整损失函数.jpg)

原文使用同一个网络来预测 Y、O 和 S，也就是说这个网络在每个位置上输出C+4个值，所有输出共享一个全卷积的backbone。

##### 测试

在测试时，我们首先提取热力图在每一类上的峰值，做法是将热力图上的所有响应点与其连接的8个临近点进行比较，如果该点响应值大于或等于其八个临近点值则保留，最后我们保留所有满足之前要求的前100个峰值点。对于每一个检测到的中心点，产生如下bbox：

![img](https://img-blog.csdnimg.cn/20190417200330996.png)

其中![img](https://img-blog.csdnimg.cn/20190417200406600.png)是偏移预测结果；![img](https://img-blog.csdnimg.cn/20190417200429484.png)是尺度预测结果。

##### 3D检测

 3D检测是对每个目标进行3维bbox估计，每个中心点需要3个附加信息：**depth, 3D dimension， orientation**。我们为每个信息分别添加head。

##### 人体姿态估计

人的姿态估计旨在估计图像中每个人的k 个2D人的关节点位置（在COCO中，k是17，即每个人有17个关键点）。因此，原文令中心点的姿态是 kx2 维的，然后将每个关键点（关节点对应的点）参数化为相对于中心点的偏移。

### 八、FCOS

论文地址：../resources/papers/FCOS Fully Convolutional One-Stage Object Detection.pdf 或 [论文链接](https://arxiv.org/abs/1904.01355)

#### 8.1 简介

FCOS全称fully convolutional one-stage object detector，它是一种不使用anchor的检测器（和Center Net的目的一样是不使用anchor），这样不仅减少了有关anchor的计算量，还不需要考虑那些anchor相关的超参。

#### 8.2 FCOS 实现

![](.\img\FCOS.jpg)

##### Fully Convolutional One-Stage Object Detector

*这一部分是FCOS最简单架构，它只包含上图的backbone和head中不包括Center-ness分支的部分。*

FCOS在特征图上的每一个点进行回归操作。首先，我们可以将feature_map中的每一个点(x,y)映射回原始的输入图片中；然后，如果这个映射回原始输入的点在相应的GT的BB范围之内，而且类别标签对应，我们将其作为训练的正样本块，否则将其作为负样本；接着，我们回归的目标是(l,t,r,b)，即中心点做BB的left、top、right和bottom之间的距离（其主要原因是为了后续使用center-ness做准备的）。FCOS使用的方法是在特征图后为分类和回归分别添加四个全卷积层，最终网络输出一个C维（这个C是类别数，在COCO数据集中为80）的分类标签和4维的bbox坐标。

由于FCOS算法是基于目标物体框中的点进行逐像素回归的，因此执行回归的目标都是正样本，所以作者使用了**exp()函数将回归目标进行拉伸**，此操作是为了最终的特征空间更大，辨识度更强。 FCOS训练的损失函数如下图所示：

<img src=".\img\FCOS损失函数.jpg" style="zoom:80%;" />

其中分类损失使用focal loss，回归损失使用UnitBox中提出的IoU损失。

##### 多尺度策略

使用FPN的策略，FCOS在不同层级的特征图上检测不同尺寸的目标，如上面的整体架构图所示，有三层是由backbone的卷积神经网络产生的，并且自顶而下连接，而另外两层则是以步长为2的下采样得到的。依据这种策略，FCOS在每一个层级的每一个像素上都进行如下操作：计算当前层级中的回归目标 l、t、r、b，判断max(l, t, r, b) > mi 或者 max(l, t, r, b) < mi -1是否满足。若满足，则**不对此边界框进行回归预测**，其中mi是作为当前尺度特征层的最大回归距离。也就是说每一层只需要回归一定尺寸的目标的像素，如果依然出现重复现象。那么就直接选择小的区域作为回归目标。

还有一点值得注意的是作者认为不同的特征层需要回归不同的尺寸范围，因此在不同的特征层使用相同的输出激活是不合理的。因此，作者没有使用标准的exp(x)函数，而是使用exp(si，x)其中si是一个可训练的标量si，能够通过si来自动调整不同层级特征的指数函数的基数。

##### Center-ness 分支

通过上述两节的方法构成的检测器和一般的anchor-based检测器精度上依然有差距，这是因为按照上文方式会产生很多由那些里目标对象中心很远的像素点产生的低质量的bbox，而Center-ness分支的提出就是为了解决这个问题。如上图所示，这个分支是一个单层分支，和分类分支并行来预测位置的Center-ness，也就是**该像素到对应目标中心的归一化距离**，其计算公式如下：

<img src=".\img\centerness.jpg" style="zoom:80%;" />

center-ness取值为0到1，训练时的损失函数为二值交叉熵损失，这一项损失也要加到上文提到的损失函数中去。在测试时，最终的得分是那一个像素的分类得分和center-ness的乘积，这样离中心远的像素的得分就会比较低，更有可能在之后的NMS中被去掉。关于这个还有一种可选方案是只选择中心点作为正样本，代价是要多一个超参。值得注意的一点是虽然看起来center-ness的计算好像可一个回归分支放在一起，但实验表明这样做效果不好。

*FCOS的思想在two-stage检测器中也可以使用，它能够用来代替RPN中的anchors*

### Notes

##### Objectness

Objectness本质上是物体存在于感兴趣区域内的概率的度量。如果我们Objectness很高，这意味着图像窗口可能包含一个物体。这允许我们快速地删除不包含任何物体的图像窗口。

##### Focal Loss

Focal loss主要是为了解决one-stage目标检测中正负样本比例严重失衡的问题。该损失函数降是在交叉熵损失上进行修改的，它低了大量简单负样本在训练中所占的权重，也可理解为一种困难样本挖掘。

##### 实例分割

由于RCNN的高效率，很多实例分割的方法是基于**分割预测**的，早期的实例分割（比如DeeoMask）是在识别之前，这样精确度低也很慢，之后出现通过bbo proposal分割的方法，但是Mask R-CNN的方法是同时预测掩码和类标签。另一种实例分割的方式是通过语义分割，就是进行像素分类，然后同类像素组合。

##### OHEM

OHEM主要思想是，根据输入样本的损失进行筛选，筛选出hard example，表示对分类和检测影响较大的样本，然后将筛选得到的这些样本应用在随机梯度下降中训练。在实际操作中是将原来的一个ROI Network扩充为两个ROI Network，这两个ROI Network共享参数。其中前面一个ROI Network只有前向操作，主要用于计算损失；后面一个ROI Network包括前向和后向操作，以hard example作为输入，计算损失并回传梯度。这种算法的优点在于，对于数据的类别不平衡问题不需要采用设置正负样本比例的方式来解决，且随着数据集的增大，算法的提升更加明显。