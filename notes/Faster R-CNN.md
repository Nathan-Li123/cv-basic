# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

*论文地址：../resources/papers/Faster R-CNN Towards Real-Time Object Detection with Region Proposal Networks.pdf*

### 一、简介

R-CNN是指 **Region-based CNN**，基于区域的卷积神经网络，因为有在不同建议之间共享卷积计算的技术，之前的Fast R-CNN在不考虑region proposal 的时候已经能够做到几乎实时的检测。目前，region proposal 是检测加速的计算瓶颈。

region proposal方法通常依赖于廉价的特征和经济的推理方案，比较典型的方法是**选择性搜索**，它基于工程化的低级特征融合超像素（依此去除掉原始图片上的冗余候选框），但是它的速度相对来说很慢。目前来说，最好地平衡了精确度和速度的是**EdgeBoxes**。如果把 proposal 阶段放到GPU上运行能提高速度，但是这样会忽略下游检测网络，从而错过了共享计算的重要机会。

Faster R-CNN 在这个问题上提出了 Region Proposal Networks (RPNs)，它可以共享最新的目标检测网络的卷积层，使 proposal 计算耗时几乎为0。Faster R-CNN 在之前的 region-based R-CNN（比如Fast R-CNN）上添加一个能够在一个**规则网格**内同时回归区域边界框和目标分数。

RPNs可以预测尺度和长宽比变化很大的region proposal。相对于使用图像金字塔和filters金字塔的方法，Faster R-CNN 提出一了种新颖的 “anchor” boxes 来作为多尺度和多长宽比的参照。可将其视为一个回归参照（regression reference）金字塔，从而避免枚举多尺度和多长宽比的图片或者filters。

文中的方法通过共享卷积特征进一步将 RPNs 和 Fast R-CNN 整合成一个网络，并提出了一种训练机制：在保持proposal固定的情况下，交替微调region proposal和object detection。

RPN 和 Faster R-CNN 框架已经被用到了很多地方，比如3D目标检测、基于部件的检测、实例分割等，同时也被 Pinterest 这类商业系统引用了。

##### region proposal

region proposal 并没有准确的中文翻译，大概意思为“预测目标大概所在的位置”，基于 region proposal 进行检测就是先大致地看一下找到大概所在的位置，然后进一步地细看。

### 二、相关工作

#### 2.1 Object Proposal

object proposal 常用技术主要有基于像素融合的SS, CPMC, MPG，基于滑动窗口的EdgeBox等。目标区域框一般被当作模型外的一部分，像基于 SS 的 R-CNN 和 Fast R-CNN等。

###### EdgeBoxes

利用边缘信息（Edge），确定框框内的轮廓个数和与框框边缘重叠的轮廓个数（这点很重要，如果我能够清楚的知道一个框框内完全包含的轮廓个数，那么目标有很大可能性，就在这个框中），并基于此对框框进行评分，进一步根据得分的高低顺序确定proposal信息（由大小，长宽比，位置构成）。而后续工作就是在proposal内部运行相关检测算法。

#### 2.2 利用深度网络进行目标检测

R-CNN 主要作为一个分类器，并不进行目标框的预测，因此他的准确度很大程度上依赖于 region proposal 模型的准确度。**OverFeat**基于全连接层对单个目标进行框坐标的预测。全连接层后接卷积层用于目标不同类别的确定。**MultiBox**方法从网络中生成区域候选框，该网络中的全连接层同时预测多个类别不确定的框。得到的类别不确定的框作为R-CNN的proposals。 而本文中使用了新 **DeepMask **的方法。使用了 Adaptively-sized pooling ( SPP ) 来提高 region-based 检测器的有效性。

#### 三、Faster R-CNN

Faster R-CNN 主要由两部分组成：一是用于生成区域候选框的深度全卷积网络，二是一个 Fast R-CNN 检测器（其实可以理解为三个部分，第一部分是共享的基础卷积层，用于提取特征，第二部分是RPN，第三部分是RoI pooling和分类网络）。整个系统是一个单一的、统一的对象检测网络，如下图所示：

<img src=".\img\Faster R-CNN 结构.jpg" style="zoom: 60%;" />

其中国 RoI pooling 为 region of interest pooling，即保留感兴趣的部分的池化。

#### 3.1 Region Proposal Networks

RPN 接受任何尺寸的图片，然后输出一些候选框以及他们的目标性评分（理解为属于各个类的评分），这整个过程是通过一个全卷积网络实现的。

首先在卷积层最后一层输出的特征图上滑动一个小的网络，该小型网络将特征图上的n×n的空间窗口作为输入，将其映射为一个更低维的特征。这个特征之后被传入两个兄弟全连接层：box-regression层（reg）和box-classification层（cls）。mini 网络执行滑动窗口形式，所有空间位置都共享全连接层。

##### Anchors

对于每个滑动窗口所在的位置，我们对多个候选区域同时进行预测，对于每个位置处，可能的proposal的最大值被记作k。因此reg层输出4 k个数据，因为需要用坐标来定位k个候选框，而cls层则输出2 k个数据，分别代表是或不是某种类别的评分。至此k个proposal就被关连至k个候选框，这个候选框就被称为anchor。默认情况下我们选用三种尺度和方向的anchor，最终得到每个位置9个锚。

###### Translation-Invariant Anchors

锚的一个重要属性是平移不变性，他的proposal计算函数也是平移不变的。如果一个对象在图片中平移了，那么相对应的锚也应该平移，并且相同的函数可以预测出任意位置的proposal。（MultiBox使用k-means算法就不具有平移不变性）同时anchor的平移不变性也能有效减小模型的大小，其参数数量比使用MultiBox的参数少了两个数量级。

###### Multi-Scale Anchors as Regression References

同时anchor的设计也提出了一种解决多尺度不同长宽比问题的新方案。传统的方案主要有两种，第一种是基于图像或特征金字塔，图片被缩放至不同尺度，特征图也在不同尺度多次计算，这种方法很耗时。第二种方法是在特征图使用不同尺度或长宽比的滑动窗口，或者说是使用过滤器金字塔，事实上第二种方法通常和第一种方法结合使用。

但是RPN使用的是锚金字塔，我们只需要一种尺度和比例的图像，也只需要一种尺度和比例的滑动窗口，这种方法更加cost-free，同时也正因为这种anchor我们才能在不需要额外定位尺度的情况下共享特征。

##### RoI pooling

RoI pooling 是 Fast R-CNN 的技术。

对于一个 H × W 的特征图区域，要将其作为输入，送入后续 h × w 的网络中，需要做shape的适配。这里适配方式就是，在 H × W 的特征图中，用 h × w 个子窗口覆盖它，每个子窗口的取值，就是对当前子窗口进行 max pooling。在RPN中会产生不同尺寸的anchor，而在之后 Fast R-CNN 的分类网络输入尺寸是固定的，所以需要 RoI层。

##### 损失函数

训练RPN时，我们为每一个anchor设置一个二进制的标签，如果anchor的IoU是最高的或者anchor的IoU超过了0.7，那么就将其标为positive，如果一个anchor的IoU低于0.3并且它不是positive的，那么就将其标为negative，不是positive也不是negative的anchor对于训练没有帮助。

Faster R-CNN的损失函数定义如下：<img src=".\img\Faster R-CNN 损失函数.jpg" alt="Faster R-CNN 损失函数" style="zoom:80%;" />

其中 i 是指在该小型网络中该anchor的序号，而 pi 是预测的该anchor为一个对象的概率，而pi加上星号则是添加的label，ti 向量anchor的四个参数化的坐标，而加上星号是相关 ground truth 框的四个参数化坐标向量。cls 的 loss 是 object 和 not object 两种类别的对数损失，而 reg 的 loss 则是使用 Smooth L1 损失函数。pi 和 ti 分别来自 reg 层和 cls 层的输出。表达式的两项分别 normalize 以先然后通过权重指数 λ 加在一起。值得注意的是上述归一化处理不是必须的，而是可以简化的。

窗口一般使用四维向量(x, y, w, h)表示，分别表示窗口的中心点坐标和宽高。偏移量计算公式如下如下：               

![](.\img\Faster R-CNN bbox 回归.jpg)

其中x，y代表候选框中心坐标，w，h分别代表宽和长，x，xa，x加星号分别代表 predicted box, anchor box, and ground-truth  box，其余变量也同理。整个过程可以理解为将predicted box 从anchor box 一步步回归为ground-truth box。

在Faster R-CNN中，用于回归的特征在特征图上具有相同的空间大小(3 3)。为了应对不同的大小，我们学习了一套边界盒回归器。每个回归器负责一个尺度和一个宽高比，这些回归器不共享权重。

##### 训练RPN

RPN使用反向传播和SGD随机梯度下降算法进行训练。

如果每幅图的所有anchor都去参与优化loss function，那么最终会因为负样本过多（背景肯定多于目标）导致最终得到的模型对正样本预测准确率很低。因此，在**每幅图像中随机采样256个anchors去参与计算一次mini-batch的损失**。正负接近比例 1 : 1 (如果正样本少于128则用负样本补全)。

###### 4-Step Alternating Training

1. backbone使用ImageNet预训练模型初始化权重，使用抽样后的256个正负例anchor框开始训练RPN网络。backbone权重也参与微调。
2. **使用第一阶段训练好的RPN，生成正例预测框**，供Fast R-CNN分类网络进行训练。此时backbone权值也使用ImageNet预训练模型初始化。截止第二步，RPN与Fast R-CNN使用两个backbone，没有共享。
3. 使用第2步中训练好的Fast R-CNN网络中相应的backbone权值，初始化RPN网络之前的backbone，RPN部分使用第1步的训练结果进行初始化。第3步只微调RPN中的权值。截止第三步，**Fast R-CNN与RPN开始共享backbone**。
4. backbone与RPN权值不再改变，使用第2步训练的Fast R-CNN部分结果初始化Fast R-CNN，再次微调训练。

#### 3.2 RPN和Faster R-CNN共享特征

Faster R-CNN 的分类网络其实就是 Fast R-CNN，首先讨论三种方式来使用共享特种训练两个网络：

* Alternation training：这种方法就是利用 RPN 生成的 proposals 去训练 Fast R-CNN，然后再用 Fast R-CNN 优化过后的网络去训练 RPN，如此不断迭代。这是原文使用的方法，交叉迭代了两次，还可以进行多次迭代，但是效果不明显。
* Approximate  joint  training：在这个解决方案中，RPN和Fast R-CNN网络在训练期间合并成一个网络，在每次SGD迭代中，前向传递生成区域提议，在训练Fast R-CNN检测器将这看作是固定的、预计算的提议。反向传播像往常一样进行，其中对于共享层，组合来自RPN损失和Fast R-CNN损失的反向传播信号。这种方法忽略了 proposal 框坐标也是一个输入函数。
* Non-approximate joint training： RoI 池接受卷积特征也接收预测边界框作为输入，因此理论上有效的反向传播求解器也应该包括关于边界框坐标的梯度。在近似联合训练方案中这些梯度被忽略了，而在一个非近似的联合训练解决方案中，就需要一个关于边界框坐标可微分的RoI池化层。

#### 3.3 实现细节

原文实现的 Faster R-CNN（可以说就是一个经典的Faster R-CNN）使用相同尺寸的图片，是将不同尺寸的图片短边缩放为600像素。在最后一层卷积层上的步长为16像素，这么长的步长已经能够让其获得很好的结果了。而对于anchors，原文使用三种面积：128平方，256平方和512平方个像素，然后三种长宽比为1 : 1，1 : 2 和 2 : 1。值得注意的是，**原文的算法允许比潜在的接受域更大的预测**（就是说接受域可能只包含了某个对象的一部分，但是 proposal 能够是包含整个对象的）。

越过**图像边界**的anchors是需要特别小心处理的，原文在训练时忽略所有越过边界的锚使他们不影响loss，不然容易引入大量难以纠正的错误项，使训练无法收敛。但是在test的时候还是不能忽略的，那时候直接将box超出部分截掉。

同时原文还引入了NMS（基于 cls 的评分），IoU 阈值设置为0.7，这基本上能使每幅图片留下大概2000个 proposal region。

### 四、Faster R-CNN 梳理

有点乱，梳理一下整个Faster R-CNN的结构。Faster R-CNN 完整结构如下图所示：

<img src=".\img\Faster R-CNN 结构02.jpg" alt="Faster R-CNN 完整结构" style="zoom:80%;" />

#### 4.1 共享卷积层部分

这部分就是应用特征提取网络，图中的13个conv层，13个relu层这个就是参考VGG16模型，也可以用其他模型，那么网络结构可能就不是这样的了，但是这个和Faster R-CNN其实没什么关系，这部分就不多写了。

#### 4.2 RPN 部分

首先需要了解重要概念anchor，见[3.1中的Anchors](#####Anchors)。

从上图中可以看到RPN网络实际分为2条线（cls 和 reg），上面一条通过softmax分类anchors获得positive和negative分类，下面一条用于计算对于anchors的bounding box regression偏移量，以获得精确的proposal。而最后的Proposal层则负责综合positive anchors和对应bounding box regression偏移量获取proposals，同时剔除太小和超出边界的proposals。到了proposal层就算是完成了目标定位。总结一下，**其实RPN最终就是在原图尺度上，设置了密密麻麻的候选Anchor。然后用CNN去判断哪些Anchor是里面有目标的positive anchor，哪些是没目标的negative anchor。**

##### cls 网络

一副M×N大小的矩阵送入 Faster R-CNN 网络后，经过 backbone之后到RPN网络变为 (M/16)× (N/16)，不妨设 W=M/16，H=N/16，首先进行一个滑动窗口的活动（活动输出过程见[3.1的第一段](####3.1 Region Proposal Networks)），注意这个滑动窗口输入和输出矩阵形状是不变的。

然后做了1×1的卷积操作，使得输出通道数为18。这也就刚好对应了feature maps每一个点都有9个anchors，同时每个anchors又有可能是positive和negative，所有这些信息都保存W×H×(9*2)大小的矩阵。

后面接softmax分类获得positive anchors，也就相当于初步提取了检测目标候选区域box。关于softmax前后的reshape原因见[notes](#####RPN中为何要在softmax前后reshape)。

##### reg 网络

首先需要掌握的是 bounding box regression 的[基本原理](#####bounding box regression)。

然后再来看 reg 网络的主线，reg 网络其实就是1×1卷积核，在这个运算中利用上文提到的[损失函数](#####损失函数)进行bbox回归。经过该卷积输出图像为W×H×36，存储为[1, 4×9, H, W]，这里相当于feature maps每个点都有9个anchors，**每个anchors又都有4个用于回归的变换量**（就是bbox回归中的d）。

##### proposal

通过 cls 那条线输出一个 W×H×18 的分类特征矩阵（完成了positive/negative的分类），通过 reg 那条线的输出一个 W×H×36 的坐标回归特征矩阵（完成了bbox回归变换量计算），之后这两个输出矩阵就进入到proposal层。Proposal Layer负责**综合所有的变换量和positive anchors，计算出精准的proposal**，送入后续RoI Pooling Layer。具体流程如下：

* 基于变换量进行bbox的具体回归操作。
* 排序 positive softmax anchors，提取前N（比如6000）个修正位置过后的positive anchor。
* 对于超出图像边界的anchor，将超出部分截掉，如果是在训练那就直接丢弃。剔除尺寸太小的anchor。
* 对剩余的anchors进行NMS操作。
* 最后输出所有留下来的anchor，也就是精确的proposals（输出形式为左上角和右下角的坐标）。

###### im_info

对于一副任意大小P×Q图像，传入Faster R-CNN前首先reshape到固定M×N，im_info=[M, N, scale_factor]则保存了此次缩放的所有信息。这也是proposal层的一个输入。

#### 4.3 Fast R-CNN 部分

##### RoI Pooling 层

而RoI Pooling层则负责收集proposal，并计算出proposal feature maps，送入后续网络。从整体结构图可以看到 RoI pooling层有2个输入：原始的feature maps和RPN输出的proposals。 

关于RoI的必要性和过程见上文的 [RoI pooling 部分](#####RoI pooling)。

##### 分类器

最后一层就是一个分类器了，准确来说它做了两件事：一是通过全连接和softmax进行分类得到结果，二是再次对proposals进行bbox回归以得到更精确的box。

### 五、训练Faster R-CNN

前文提到 [Faster R-CNN 的训练是四步](######4-Step Alternating Training)的（原文的训练方法，也有其他训练方法），下图是一张训练流程图（*图中的Faster R-CNN应该是Fast R-CNN*）：

<img src=".\img\Faster R-CNN 训练流程图.jpg" style="zoom:90%;" />

首先使用训练好的model（例如VGG、ZF），整个网络（只到RPN）的 Loss 使用见上文[损失函数](#####损失函数)，然后反向传播进行迭代训练。然后利用训练好的RPN网络获取porposals，保存起来，然后传入 Fast R-CNN 部分去训练 Fast R-CNN 网络。训练好之后将 Fast R-CNN 和 RPN 给连起来，再利用现在的 Fast R-CNN 第二次训练 RPN，此时只更新 RPN 的网络层。最后再一次和刚刚一样第二次训练 Fast R-CNN。

### 六、Notes

##### RPN中为何要在softmax前后reshape

只是为了方便softmax分类。输入矩阵存储形式为[1, 2×9, H, W]，**softmax分类时需要进行positive/negative二分类**，所以reshape layer会将其变为[1, 2, 9×H, W]大小，即单独“腾空”出来一个维度以便softmax分类，之后再reshape回复原状。

##### 超参

超参数其实就是一个参数，是一个未知变量，但是它不同于在训练过程中的参数，它是可以对训练得到的参数有影响的参数，需要训练者人工输入，并作出调整，以便优化训练模型的效果。

##### 关于class-aware和class-agnostic

class-aware的输入是一张图片，返回的是对应需要检测的每一个类的bbox,即检测器在完成检测的同时也知道所检测的每一个bbox属于哪一类，而class-agnostic的输入是一张图片，返回的是一组对象，但是并不知道这组对象中每一个具体属于哪一类，即只能进行前景与背景的检测。

##### train，val 和 test

train和test很好理解，一个是指训练数据集而另一个测试数据集。

val是validation的简称。training dataset和validation dataset都是在训练的时候起作用，而因为validation的数据集和training没有交集，所以这部分数据对最终训练出的模型没有贡献。validation的主要作用是来验证是否过拟合、以及用来调节训练参数等。

##### bounding box regression 原理

box 的坐标一般用（x，y，w，h）表示，分别代表中心坐标、宽和长。bbox 回归的目标是寻找一种关系，使得输入原始的anchor A经过映射得到一个跟真实窗口G更接近的回归窗口G'。这种变化比较简单的实现就是先平移再缩放：

- 先做平移

![[公式]](https://www.zhihu.com/equation?tex=G_x%27%3DA_w%5Ccdot+d_x%28A%29+%2BA_x%5Ctag%7B2%7D)

![[公式]](https://www.zhihu.com/equation?tex=G_y%27%3DA_h%5Ccdot+d_y%28A%29+%2BA_y%5Ctag%7B3%7D)

- 再做缩放

![[公式]](https://www.zhihu.com/equation?tex=G_w%27%3DA_w%5Ccdot+%5Cexp%28d_w%28A%29%29+%5Ctag%7B4%7D)

![[公式]](https://www.zhihu.com/equation?tex=G_h%27%3DA_h%5Ccdot+%5Cexp%28d_h%28A%29%29%5Ctag%7B5%7D)

可以看到其实需要学习的就是四个变换，当输入的anchor A与GT相差较小时，可以认为这种变换是一种线性变换， 那么就可以用线性回归来建模对窗口进行微调。线性回归就是给定输入的特征向量X, 学习一组参数W, 使得经过线性回归后的值跟真实值Y非常接近。对于bbox回归问题，输入 X 是 CNN feature map，定义为Φ；同时还有训练传入A与GT之间的变换量或者说差值，就是四个t值。那么目标函数可以表示为：

![[公式]](https://www.zhihu.com/equation?tex=d_%2A%28A%29%3DW_%2A%5ET%5Ccdot+%5Cphi%28A%29%5Ctag%7B6%7D)

其中的星号就是x，y，w，h中的一个。到这里就清晰了，我们要做的就是调整参数，也就是上面公式中的W，使我们的预测值 d 和 真实的偏差值 t 尽可能地接近，这里就需要用到损失函数了。

##### anchor free 和 anchor based

这是目标检测算法的一种分类方式（可类比One-Stage和Two-Stage分类），区别就在于**有没有利用anchor提取候选目标框**。anchor based 的典型算法有 Faster R-CNN、SSD、YOLO等，anchor-free类算法代表是CornerNet、ExtremeNet、CenterNet、FCOS等。

### 七、PyTorch 实现

关于使用PyTorch实现Faster R-CNN另写了一篇记录，详见[Faster R-CNN实现](./PyTorch Faster R-CNN.md)。







