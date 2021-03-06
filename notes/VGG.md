# VGG

论文地址：../resources/papers/Very Deep Convolutional Networks for Large-Scale Image Recognition.pdf

### 一、简介

之前ILSVRC-2013中最好的提交利用的是更小的感受野和更小的步长，还有一种方式是密集地在多尺度的整个图像上进行训练和测试，而本文针对的是CNN的另一个方向——深度。简单思路就是稳定地增加卷积层，这种操作因为每个卷积层的卷积核都是很小的3×3而可行。

VGG是Oxford的**V**isual **G**eometry **G**roup的组提出的（这也是VGG这个名字的由来）VGG有两种结构，分别是VGG16和VGG19，两者并没有本质上的区别，只是网络深度不一样。

### 二、卷积网络配置

#### 2.1 VGG通用架构

训练时输入网络的是固定的224×224的RGB图像，这些图像只通过一种预处理：**每个像素减掉整幅图像的RGB平均值**。然后图像经过卷积层，在每个卷积层中VGG设置的滑动窗口都是3×3*的，除此之外还会有一个1×1的过滤器，它可以被视为是对输入的通道数进行线性变换。卷积层步长被设置为1，3×3卷积核的padding也是1pixel。同时内部含有五个max pooling层，其尺寸为2×2，步长也为2pixels。在这些卷积层之后（不同的配置卷积层数不一样）是三个全连接层：前两个有4096个通道，而最后一个有1000个通道。在之后就是最后一层，softmax。

注意VGG网络没有使用  Local  Response  Normalisation(LRN) ，局部响应归一化，作者的理由是这对ILSVRC数据集测试没有帮助，反而会导致内存消耗和计算时间的增加。反正事实就是到现在大部分VGG都是没有LRN的，但也有加上了LRN层的VGG。

#### 2.2 VGG配置

就像我之前提到的VGG有不同的配置。这个配置的区别就体现在深度上，从11层（8个卷积层加3个全连接层）到19层（16个卷积层加3个全连接层），目前比较常见的是VGG-16和VGG-19。卷积核的通道数从一开始的64每经过一个max pooling翻个倍，直到512为止。

#### 2.3 优点

VGG使用了多层3×3卷积层来代替大的卷积核，比如5×5或者7×7，那么这么做有什么优势呢？

* 分成多层以后非线性的激活函数变多了，这使得分类器更具有识别力。
* 减少了参数使得计算变快。
* 验证了通过不断加深网络可以提升性能。

PS：事实上VGG消耗更多的计算资源，尽管使用了3×3卷积，但绝大多数参数来自全连接层，导致参数很多。

### 三、分类框架

#### 3.1 训练

VGG网络的训练是通过使用**小批量梯度下降**来进行的，基于**有冲量的反向传播**（batch_size 是256，momentum设置为0.9）来优化多项 logistic regression（*注意名字太多了别搞混了，逻辑回归损失函数就是对数损失函数，也就是交叉熵损失*）。训练通过权重衰减（L2惩罚乘子设定为）进行**正则化**（关于正则化见Notes中的[正则化](#####正则化)），前两个全连接层执行丢弃正则化（丢弃率设定为0.5）。学习率初始设定为0.01，然后当验证集准确率停止改善时，减少10倍。学习率总共降低3次，学习在37万次迭代后停止（74个epochs）。

网络权重的初始化是非常重要的，因为深度网络中梯度不稳定，不好的初始化可能会阻碍学习。为了规避这个问题，VGG初始化时首先训练最浅的11层VGG，因为他比较浅所以可以随机初始化然后训练。当我们需要训练更深的VGG的时候，前四层和后三层我们直接使用已经训练好的11层VGG的相应层进行初始化，而其他层随机初始化。当需要随机初始化时，VGG从一个均值为零、方差为10的负二次的正态分布中取样权重，而偏置值为0。还有一种方法是使用 Glorot＆Bengio（2010）的随机初始化程序来初始化权重而不进行预训练。

训练时图片预处理的流程是首先得到一个训练尺度S值，怎么得到就是下文sigle-sacle和multi-scale两种方法。第二步是对图像进行等比例变换使得图片的短边等于S，然后进行随机剪裁（剪成224×224，也因此S必须大于等于224）。为了进一步增强数据集，最后对图像进行水平翻转和随机RGB颜色偏移（原文用的应该是减去ImageNet训练集的RGB均值）。

###### single-scale training

固定S的值，原文中使用的是256和384，为了加快速度先训练S=256，而S=384的训练时基于之前的进行，同时学习率更低（0.001）。

###### multi-scale training

每个训练图像从一定范围内选择S的值，比如原文用的就是[256，512]。

#### 3.2 测试

在测试时，给定一个经过训练的ConvNet和一个输入图像。首先预定义一个虽小图像边 Q，这里的Q不一定要等于 S，事实上每个S使用多个Q能够改进性能。然后将输入图像等比例缩放为最短边长度为Q并传入卷积网络。测试时，第一个全连接层被转化为7×7的卷积层（padding为0，stride为1），而后两个被转化为1×1的卷积层（为什么要这么做见[Notes相应部分](#####VGG网络中测试时为什么全连接层改成卷积层)，然后得到的完全卷积的网络被应用到图像上（未剪裁的）。得到的结果是一个类得分图和一个取决于输入图像大小的可变空间分辨率。最后进行一个sum pooling来得到一个固定的类别分数向量。最后还要通过水平翻转图像来增强测试（就是反转一下输入图像然后再跑一遍刚刚的流程），然后将原始图像和反转图像结果softmax之后进行平均得到最终结果。

### 四、VGG定位

#### 4.1 定位卷积网络

卷积网络和之前的分类网络一样，只不过最后的全连接层不是给出每类的分数而是预测bbox位置（用中心坐标和长宽来表示）。bbox是所有类通用还是每个类有各自的框是可选的。

#### 4.2 训练

定位网络的训练和分类网络的训练很像，主要差别就是将交叉熵损失换成了欧几里得损失，后者能够惩罚bbox的偏移。

#####  Euclidean Loss

就是L2范数损失函数，实际上是计算预测值到目标值的欧几里得距离的平方（没必要再开个根号算距离，会增大计算负担）。

#### 4.3 测试

主要有两种测试方式：第一种是在val上比较网络改进，只考虑ground truth类的边界框预测(不考虑分类错误)。边界框是通过只对图像的中心裁剪应用网络得到的。而第二种则是完整的测试，和分类测试差不多，区别在于不输出类别分图而是一组bbox预测。VGG的边界框预测使用 greedy merging procedure：先融合空间上相近的边界框，然后用他们的分类分数评估他们。

### 五、Generalization of Very Deep Features

这一节主要就是讲将只应用VGG的特征提取部分，然后将预训练好的（原文中使用的是ILSVRC数据集训练的）VGG模型结合其他的分类器进行测试。应用时将VGG的最后一个全连接层，也就是那个输出1000个分类的分类器去掉，将倒数第二层输出的4096个channel激活后作为提取的特征输出。输出之后进行max pooling，得到一个4096维的**特征描述子**，这些描述子之后还要和水平反转图像的描述子平均一下。和完整的测试一样，这种情况的VGG也可以有多个Q，多个尺度得到的结果可以堆积（stack）起来也可以池化一下。堆积起来可以帮助分类器学习如何最优地结合不同尺度的图像数据，但是会增加描述子维度。

### 六、Notes

##### Top-1 和 Top-5

Top-1错误率：对一个图片，只判断概率最大的结果是否是正确答案。Top-5错误率：对一个图片，判断概率排名前五中是否包含正确答案。

##### 正则化

正则化是解决高方差问题的重要方案之一，也是Reducing Overfiltering（克服过拟合）的方法。过拟合一直是DeepLearning的大敌，它会导致训练集的error rate非常小，而测试集的error rate大部分时候很大。

* L2正则化：L2正则化倾向于使网络的权值接近0。这会使前一层神经元对后一层神经元的影响降低，使网络变得简单，降低网络的有效大小，降低网络的拟合能力。L2正则化实质上是对权值做线性衰减，所以L2正则化也被称为权值衰减（weight decay）。

* 失活（dropout）：最常用的是随机失活，即在训练时随机选择一部分神经元将其置零，不参与本次迭代，它能够降低神经元之间的耦合。dropout一般在全连接层之后，每次forward的时候全连接层之前的神经元会以一定概率不参与本次迭代，不过这样也会一定程度延长拟合时间。**inverted dropout已经被理论上证明等价于L2正则化了**

* 数据扩充：这实质是获得更多数据的方法。当收集数据很昂贵，或者我们拿到的是第二手数据，数据就这么多时，我们从现有数据中扩充生成更多数据，用生成的“伪造”数据当作更多的真实数据进行训练。

* 早停：当验证集误差不再变化或者开始上升时提前停止训练。常见的做法是，在训练的过程中，记录最佳的validation accuracy，当连续10次epoch（或者更多次）没达到最佳accuracy时，你可以认为“不再提高”，此时就停止训练。

  ***有一点要注意的是正则化方法可以同时应用不止一种***

##### VGG网络中测试时为什么全连接层改成卷积层

卷积层和全连接层的唯一区别在于卷积层的神经元对输入是局部连接的，并且同一个通道(channel)内不同神经元共享权值(weights)。卷积层和全连接层都是进行了一个点乘操作，它们的函数形式相同. 因此卷积层可以转化为对应的全连接层，全连接层也可以转化为对应的卷积层。

例如原文的VGGNet第一个全连接层的输入是7×7×512，输出通道数为4096，全连接层的做法是生成一个权重矩阵（共7×7×512×4096个权重参数），我们可以用4096层的7×7的卷积核替代（可以看到参数数量是一样，因此可以直接套用之前训练得到的参数），这样就得到了一个4096×1×1的矩阵，同理后两层就可以转换为1×1卷积核。

将全连接层改为卷积层原因是VGG的**test和train的图像大小是不一样的**，全连接层的权重矩阵经过训练之后已经固定下来，如果改变了输入大小，全连接层是会报错的，而转换为卷积层之后，因为卷积核是滑动的，他对输入尺寸没有要求。只不过224×224图像的输出是1×1×1000，而其他尺寸的输出是a×b×1000。

##### Caltch 数据集

加州理工大学图像数据库，~~这个数据集的评估方法是随机的生成一些splits放到训练和测试数据集中，然后返回这些splits的平均识别率~~（用mean class recall表示）。理解错了，不是生成一些splits，只不过是它不划分训练集和测试集，需要自己去选择一些图片作为训练集，另一些作为训练集。







