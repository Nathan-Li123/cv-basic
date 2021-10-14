# Object Detection in 20 Years 

### 一、目标检测发展过程

*以下算法都是具有里程碑意义的*

#### 1.1 传统算法

##### 主要算法

VJ Detecor - HOG Detector - DPM

##### DPM

DPM是传统算法的**巅峰**，是VOC-07，08，09目标检测比赛的赢家，它是HOG方法的拓展。尽管现在的目标检测算法远远强过了DPM，但是DPM提出的很多东西，现在都在沿用，例如难例挖掘，Bbox 回归。

#### 1.2 One-stage 算法

##### 主要算法

YOLO - SSD - RetinaNet

#### 1.3 Two-stage 算法

##### 主要算法

RCNN - SPPNet - Fast RCNN - Faster RCNN - FPN

***One-stage算法一步到位，直接定位目标，速度更快，Two-stage算法生成候选框然后分类、修正，最后得出结论，精准度高但慢一些***

### 二、目标检测中的技术发展

#### 2.1 早期传统方法

在2000年之前，没有一个统一的检测哲学，检测器通常都是基于一些比较浅层的特征去做设计的。

#### 2.2 早期CNN检测技术

最早在1990年，杨乐春(Y.LeCun)就已经开始使用CNN做目标检测了，只是由于当时的硬件计算力水平不行，所以导致设计的CNN结构只能往简单的去设计，不然没有好的硬件去运行完成实验。

#### 2.3 多尺度检测的技术发展

##### 时间轴

特征金字塔加滑动窗口 - 先使用object proposal然后检测 - 使用deep regression - 使用Multi-reference detection

##### object proposal

这种方法一般是将检测分为两个部分，第一个部分先做推选框，第二部分是根据推选框做进一步的分类，基于推选框的算法，一般有以下几个特征：1、召回率比较高；2、时间消耗比较大。

##### Multi-reference detection

对于检测多尺度目标，目前最流行的方法还是Multi-reference，其主要的思想就是预先定义一组reference boxes，例如经常用的anchor box，它们具有不同的尺寸和缩放因子，然后检测器基于这些boxes，去做运算。这种类型算法的loss一般是由两部分组成，第一个是定位loss，另一个是分类loss，两者加权求和之后就是最后的目标检测的loss

#### 2.4 Bounding Box Regression技术的发展

bbox回归对于目标检测的定位精度的提升至关重要，它主要是为了修正基于proposals的bbox的位置。

##### 时间轴

无BBox回归 - 从特征图得到BBox

#### 2.5 NMS技术发展

nms是一个非常重要的技术手段。如果对于有同一个目标上出现多个检测的框的时候，NMS可以根据每个框的score来进行优化，去除掉一部分的多于的框。

##### Greedy Selection

Greedy Selection 的思路很简单，就是选取得分最高的一个检测框，而他周围的检测框按照预设的重叠阈值移除。

##### BB aggregation

BB aggregation 是一组NMS技术，其主要思想是将多个有重叠的检测框合成一个检测器，这种方法的好处是能够充分考虑物体之间的关系以及他们的布局，VJ detector 和 the Overfeat 都使用了这种方法。

##### Learning to NMS

这类方法的主要思想是把NMS看作一个过滤器来对所有检测器重新打分，并且以端到端的方式将NMS作为网络的一部分进行训练，这种方式的准确度很高。

#### 2.6 困难样本挖掘的技术发展

##### 困难样本

 在目标检测深度学习的训练过程中，正负样本的比例其实不均衡的，因为标注的数据就是正样本，数量肯定是固定的，为了保证正负样本的均衡，所以会选取一定的背景图片作为负样本，但是背景图片的样本集是一个open-set，不可能全部参与训练。所以需要将训练过程中难以训练的样本挖掘出来，给以更高的loss来训练，促进模型的泛化能力。

##### 时间轴

Bootstrap - without Hard Negative Mining - Bootstrap + New Loss Functions

### 三、目标检测的加速

#### 3.1 加速技术分类

speed up of detection pipeline, speed up of detection engine, speed up of numerical computations

#### 3.2 常见加速技术

##### 特征图共享计算

Feature Map Shared Computations，在目标检测算法中，特征提取阶段往往耗时往往最多。在特征图共享计算里面分为两种，第一种是空间计算冗余加速，第二种是尺度计算冗余加速。

##### 分类器加速

早期目标检测中，是提取特征加上分类器这样一个套路来进行目标检测的，分类器一般是线性分类器，但是线性分类器没有非线性分类器效果好，例如svm就是非线性的，所以加速分类器的运行也是提升检测算法速度的一个方法。

##### 级联加速器

*cascade Detection*

其主要思想是使用简单计算器过滤掉大部分的背景检测框，然后再使用复杂的计算器来计算那些复杂的检测框。级联检测器可以很好的将计算耗时固定在一个比较小的范围，采用多个简单的检测，然后将其级联，从粗到细的过滤。

##### 网络剪枝和量化

* Network Pruning：网络剪枝，在原来网络结构的基础上，对于一些网络结构进行修剪，在尽量不影响精度的前提下降低网络的计算量，例如减少通道数，合并网络层参数等等。
* Network Quantification：网络量化，通过例如将原来浮点数计算量化为定点计算，甚至于变为与或运算来降低网络的运算量
* Network Distillation：将一个比较复杂的网络的学习到的“知识”蒸馏出来，“教给”一个比较小的网络学习，这样小网络的精度比较高，运算耗时也比较小。

##### 轻量级网络设计

轻量级网络设计( Lightweight Network Design )目前最热门的加速方式，常见的mobileNet的设计就是这个轻量级网络设计的典型代表。

* 分解卷积，将大卷积核分解为几个小的卷积核，这样其运算参数量就会降低。例如一个7x7的卷积核可以被分解为3个3x3的卷积核，它们的感受野相同，计算量后者要小，例如一个kxk的卷积核可以被分解为一个kx1和一个1xk的卷积核，其输出大小也相同，计算量却不同
* 分组卷积，在早期硬件显存不够的情况下，经常用分组卷积来进行降低计算量，将特征通道分为不同的n组，然后分别计算
* Depth-wise Separable Conv，深度可分离卷积，首次是mobileNet中提出来的，大大降低了卷积过程中的计算量。将普通卷积的深度信息分离出来，然后再利用1x1卷积将维度还原，即降低了计算量又在一定程度上使得特征图的通道重组，加速效果非常好
* Bottle-neck Design，经常被用在轻量级网络的设计上，例如mobileNetV2就使用了反瓶颈层去设计网络。
* Neural Architecture Search，简称NAS，从2018年AutoML问世以来，NAS发展非常的火，这种小型的网络结构是被训练自动搭建出来的。给机器限定一个搜索空间，让机器自己学习搭建一个高校的网络，总目前的效果来看，NAS搜索出来的结构比人工搭建的网络更加高效。例如mobileNetV3，efficientNet。

##### 数学加速

积分图像加速、频域加速、矢量量化、Reduced Rank Approximation

### 四、目标检测近期发展

#### 4.1 使用更好的引擎

引擎指的是特征提取的主干网络。

##### 主要实例

AlexNet、VGG、GoogLeNet、Resnet、DenseNet、SENet

#### 4.2 使用更好的特征

##### Feature Fusion 特征融合

同变性（Equivariance）和不变性（Invariance）是图像特征表达的两个重要指标，同变性在学习语义信息表示的时候非常重要，但是在目标定位的时候不变性又变得非常重要，所以往往需要进行特征融合。在池化的深层CNN中，深层特征不变性强，有助于分类但是不利于定位，而浅层的特征则相反，因此需要深浅层的特征进行融合。

* Processing flow：似于SSD的架构那种，将不同层次上的特征图进行融合，以适应不同大小目标的检测，使用跳跃链接引出然后融合特征。
* Element-wise operation：此种方法非常简单，就是将特征图中的每一个元素进行简单的相加，相乘操作，暴力的进行融合。

#####  高分辨率特征

还有另外一种更好的特征表达方式，那就是增大特征图的分辨率，也就是说特征图在原图上有着更大的感受域，这样对于检测能力也有非常的提升。有一种同时提高分辨率和接受域的方法是使用Dilated convolution，其主要思想是扩大卷积滤波器并使用稀疏的参数。

#### 4.3 不止于滑窗

##### Detection as sub-region search

将检测过程视为寻找从初始网格到覆盖所有所需网格的路径的过程，或者认为检测是一个迭代更新过程，以细化预测的边界框的角的过程

##### Detection as key points localization

将目标检测看成是关键点检测的问题，因为一个目标可以被表示为左上角和右下角的坐标包围的矩形框，所以这类问题可以被转换成不依赖于anchor的定位问题。

#### 4.4 目标定位能力提升

##### Bounding Box Refinement

Bounding Box Refinment 会不断将检测结果反馈给 BB regressor 直到预测结果覆盖为正确的地点和大小。但是也有一些研究者认为反复进行 BB Regression 可能会使结果更差。

##### 改善损失函数

因为目前大部分的损失函数设计都是通过计算IoU来得到定位的loss，这样对于end2end的思想还是相差的有点远，如果能够重新设计一个loss函数来更好的表示定位误差，这样训练过程会更加的好。

#### 4.5 Learning with Segmentation

在训练过程中，我们标注的都是矩形框，矩形框中或多或少都会标有一部分背景信息，如果没有语义信息，那么这种训练其实是不完美的。甚至于有些目标的外形比较奇怪，例如一个猫和一个非常长的火车，如果计算IoU的话，这样计算结果就不能很好的表示定位误差。如果带有语义信息的训练，然后使用多任务的损失函数，这样可以帮助到网络进行很好的学习。

* Learning' with enriched features：将语义分割的网络作为额外的特征加入检测框架，不过会导致计算量增加
* Learning with multi-task loss functions：在检测框架中增加一个语义分割分支，用多工作损失函数（segmenta-tion loss + detection loss）来训练吗模型

#### 4.6 旋转和尺度变化的健壮检测

##### Rotation Robust Detection

传统的解决物体旋转检测的方法一般是增大数据集使物体的每个方向都被覆盖或者为不同方向建立多个检测器。现在有了一些新的方法：

* Rotation invariant loss functions
* Rotation calibration：主要思想是对目标检测的候选框进行几何变换来适应角度的变换
* Rotation RoI Pooling：通常情况下目标物体的特征连成网络是在笛卡尔坐标系下的，所以他在旋转时不是不变的，最近有一种改进使特征连网在极坐标系下进行，这时候这些特征就不受旋转影响了。

##### Scale Robust Detection

* Scale adapted training：在训练和检测的时候都建立图像金字塔，并且只回传一定尺度图像的损失。
* Scale adapted detection：自动的放大某些小的目标图像或者学习预测物体在图像中的大小分布，然后自适应地调整图片大小。

#### 4.7 Training from scratch

目前绝大部分检测器都是先使用大量数据集进行训练然后再在检测的时候调整模型，但最近有些研究者开始尝试通过scratch(*应该是指随机数据？*)训练检测器。

#### 4.8 对抗训练

The  Generative  Adversarial  Networks  (GAN)  含有两个神经网络，生成网络生成拥挤的对象掩模，直接在特征层缩小拥挤程度，造成对抗攻击，而鉴别器网络需要鉴别真实数据分布和生成器网络产生的数据。将GAN思想应用到目标检测中，特别是可以提高小目标和重叠目标的检出率，也能增强检测器的可靠性。

#### 4.9 弱监督学习

Weakly Supervised Object Detection (WSOD) ，弱监督学习。一般的检测器训练都需要给检测目标图片打上大量的标签，弱监督学习就是试图使用只有图片级别注解而没有bbox层的标注的数据集训练检测器。主要方法有multi-instance learning、class activation mapping等。

### Q & A

1. Q : by detecting objects of different scales **at different layers of the network**. 从网络的不同层检测什么意思？

   A : 

2. Q : NMS 的 greedy selection 中说 the top-scoring box may not be the best fit，为什么？

3. Q : p 15, **It  takes  a  coarse  to  fine  detection philosophy**, 啥意思？

4. Q : detectors中的 filter, channel, layer 分别指的是什么？

