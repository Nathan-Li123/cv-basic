# FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking

论文链接：[FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking](https://arxiv.org/abs/2004.01888)

### 一、简介

目前主要的MOT检测器是使用两个独立的模型：先使用检测模型在每个framework上获取bbox，然后使用关联模型将每个framework上的bbox进行关联。随着two-step方法的成熟，更多的研究人员开始研究同时检测目标和学习Re-ID特征的one-shot算法，当特征图在目标检测与Re-ID之间共享之后，可以大大的减少推理时间，但在精度上就会比two-step方法低很多，作者分析的三个原因如下：

##### Anchor-Based检测器不适合提取re-ID特征

anchor-based检测器首先生成anchor来检测目标，然后从检测结果中提取re-ID特征，因此在训练时会把重心放在生成精确的目标proposals上（如果目标检测就是错误的那么re-ID特征将毫无意义）。同时这样会引入模糊性，一个anchor可以对应多个re-ID特征或者多个anchor对应一个re-ID特征（特别是在人群中这种情况）。还有一点是实际物体的中心可能与负责对该物体进行检测的anchor中心有偏差。

##### 多层特征共享问题

anchor进行目标检测的主要进行的是不同分类的区分，而re-ID检测注重的是个体的区分，他们关注的特征层级不太相同，re-ID更关注低级特征。

##### 特征维度问题

之前的re-ID的特征维度通常为512甚至1024，这远比目标检测的特征维度要高，这种差异会影响检测器的表现。而实验表明，实际上在MOT中使用低一些维度的特征反而效果更好。

### 二、FairMOT架构

为了解决上述问题，作者基于CenterNet提出了Fair MOT，这种检测器同等对待detection和re-ID两个任务。其总体架构如下图所示：

<img src="..\img\Fair MOT架构.jpg" style="zoom:80%;" />

#### 2.1 backbone

原文中作者使用了ResNet-34来作为backbone，这样能比较好的平衡精确度和速度，同时还使用了一个加强的Deep Layer Aggregation（DLA）来融合不同层的特征。与原本的DLA不同的是在低层和高层之间添加了更多的连接(与FPN相似)，另外在上采样的时候采用的是可形变卷积，可以根据目标的尺度和姿势动态的适应感受野，这个改进也有助于缓解对齐问题。最终得到的模型被称为DLA-34，其结构如上图那个Encoder-decoder network部分所示，下采样stride为4。

#### 2.2 Detection Branch

与centernet一样，作者将目标检测看作是高分辨率特征图上基于中心的边界盒回归任务。三个平行回归head被添加到backbone中，分别用来预测heatmap、对象中心偏移量和box大小。每个head的实现方法是对backbone的输出特征图进行3×3卷积(256通道)，再经过1×1卷积层生成最终目标。

##### Heatmap Head

这个head负责预测物体中心的位置。本文采用了基于heatmap的表示，它是关键点预测任务的实际标准。heatmap的大小为H * W * 1，如果热图中的某个位置与标签物体中心坍塌，则该位置的响应预计将是一致的。随着热图上的位置与物体中心之间的距离，响应呈指数衰减。具体实现见[Classic Algorithms](../Classic Algorithms.md)的CenterNet相关部分，训练时使用带facal loss的逻辑回归。

##### Center Offset Head

这个head负责更精确地定位对象。feature map的步长是4，这将引入不可忽略的量化误差。注意，这对目标检测性能的好处可能是边际的。但这对于跟踪是至关重要的，因为Re-ID特征需要根据准确的目标中心提取。作者在实验中发现，ReID特性与对象中心的仔细对齐对性能至关重要。

##### Box Size Head

这个head负责估计每个锚点处目标边界框的高度和宽度。该头部与Re-ID特征没有直接关系，但定位精度将影响目标检测性能的评价。上述的两个head的损失函数都使用L1损失。

#### 2.3 Identity Embedding Branch

身份嵌入分支的目标是生成能够区分不同对象的特征。理想情况下，不同物体之间的距离应该大于同一物体之间的距离。为了实现这一目标，作者在backbone特征图上应用一个有128个核的卷积层来提取每个位置的身份嵌入特征，得到128×W×H的feature map，一个(x, y)上的Re-ID特征向量就是来自这个feature map。FairMOT将对象识别嵌入作为分类任务，训练集中具有相同标识的所有对象实例（也就是不同图像中的同一对象）都被视为一个类。对于图片中的每一个标签框，在heatmap上获得目标中心(cxi, cyi)，提取一个恒等特征向量Exi,yi定位并学习将其映射到一个类分布向量p(k)，表示标签的one-hot编码为Li(k)。其损失函数如下：

<img src="..\img\Re-ID损失.jpg" style="zoom:60%;" />

其中K是训练集中身份嵌入的总数。

#### 2.4 训练

训练时总的损失函数如下：

<img src="..\img\FairMOT损失函数.jpg" style="zoom: 80%;" />

其中w1和w2是可学习的参数，用来平衡两个分支的损失。

#### 2.5 测试

网络接受1088×608的frame作为输入，得到heatmap之后做一个3×3的max pooling操作来进行NMS，并留下热力图得分大于一个阈值的点，针对这些进行center offset和size的回归并提取响应的身份嵌入。

关于关联方法则采用了hierarchi-cal online data association method。首先在第一个frame上基于bbox定义一定数量的tracklets，然后使用two-stage的方法在后续frame中使用bbox去链接这些tracklets。第一步使用卡尔曼滤波器和re-ID特征来获取初始的tracking结果，第二部对于未匹配的检测和tracklets，基于他们的box重合率来进行链接。最终还需要为一些无法匹配的bbox建立新的trackltes，并将无法找到匹配的tracklets保留30个frames以免该对象再次出现。

















