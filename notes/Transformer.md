# Transformer

### 一、NLP 部分

transformer模型最早是在自然语言处理领域提出的，论文链接：[Attention is All You Need](http://arxiv.org/abs/1706.03762)

#### 1.1 简介

在此之前RNN是语言建模、机器翻译等领域最常用的检测器方法，它非常擅长处理这些input是sequence，但是它有个问题，不容易被平行化(单向信息流)，也就是说每个word Embedding都不能同时计算，而需要按照顺序执行。之后出现了注意机制（attention），并将其应用到机器翻译等领域，取得了很不错的效果。而Transformer则是一个完全抛弃了RNN而基于Attention Machanism的模型架构，它能够很好的解决并行计算问题。

解决该问题还有一个方法是使用卷积神经网络，比如 ByteNet 和 ConvS2S，但是采用这种方法对于两个随机输入输出位置的计算量或操作数会随着两者的位置变化而变换，但是Transformer的计算量则是一个固定值。

#### 1.2 Transformer 架构

绝大部分的序列处理模型都采用**encoder-decoder结构**，其中encoder将输入序列映射到一个连续表示上，然后decoder生成一个输出序列 ,每个时刻输出一个结果。Transformer模型延续了这个模型，整体架构如下图所示：

<img src=".\img\transformer01.jpg" style="zoom:70%;" />

##### encoder & decoder

首先了解一下encoder-decoder框架，这个框架之前最经典的是使用RNN实现的。编码阶段由编码器将输入序列转化为一个固定维度的稠密向量，解码阶段由解码器将这个激活状态生成目标译文。通俗的讲，我们大脑读入的过程叫做Encoder，即将输入的东西变成我们自己的记忆，放在大脑当中，而这个记忆可以叫做Context，然后我们再根据这个Context，转化成答案写下来，这个写的过程叫做Decoder。其实就是编码-存储-解码的过程。

然后再看Transformer的encoder-decoder，encoder由六个相同的层组成，而每一层又有两个子层，如上图左侧的那个标着N×的模块所示，第一个子层是一个multi-head的self-attention机制，而第二子层则是一个简单的前馈全链接层，同时对于每个子层前后还使用了残差网络和归一化。为确保连接，所有的层输出维度都是512。decoder也是六层，如上图右侧模块所示，它每一层包含三个子层，第一层是一个Masked multi-head self-attention，而后两层和encoder中的两个子层一样，并且和encoder一样应用了残差网络和归一化。其输入输出和解码过程如下：

- 输出：对应i位置的输出词的概率分布
- 输入：encoder的输出和对应i-1位置decoder的输出。所以中间的attention不是self-attention，它的K，V来自encoder，Q来自上一位置decoder的输出
- 解码：这里要注意一下，训练和预测是不一样的。在训练时，解码是一次全部decode出来，用上一步的ground truth来预测（mask矩阵也会改动，让解码时看不到未来的token）；而预测时，因为没有ground truth了，需要一个个预测。

##### Attention Machanism

Attention用于计算"相关程度", 例如在翻译过程中，不同的英文对中文的依赖程度不同，Attention通常可以进行如下描述，表示为将query(Q) 和键值对 <img src="https://www.zhihu.com/equation?tex=%5Cleft%5C%7B+K_i%2CV_i%7Ci%3D1%2C2%2C...%2Cm+%5Cright%5C%7D" alt="[公式]" style="zoom:80%;" /> 映射到输出上，其中query、每个key、每个value都是向量，输出是V中所有values的加权，其中权重是由Query和每个key计算出来的。计算方法分为三步：

第一步：计算比较Q和K的相似度，用f来表示

![[公式]](https://www.zhihu.com/equation?tex=f%28Q%2C+K_i%29%2C+i%3D1%2C2%2C...%2Cm%5Ctag%7B1%7D)

第二步：将得到的相似度进行Softmax操作，进行归一化

![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_i+%3D+%5Cfrac%7Be%5E%7Bf%28Q%2CK_i%29%7D%7D%7B%5Csum_%7Bj%3D1%7D%5E%7Bm%7D%7Bf%28Q%2CK_j%29%7D%7D%2C+i%3D1%2C2%2C...%2Cm%5Ctag%7B2%7D)

第三步：针对计算出来的权重 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha_i) ,对V中所有的values进行加权求和计算，得到Attention向量

![[公式]](https://www.zhihu.com/equation?tex=%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%7B%5Calpha_iV_i%7D%5Ctag%7B3%7D)

Transformer中使用的注意机制是Scaled Dot-Product Attention（就是上面所说的Attention操作的的第一个公式使用的是点乘计算，不过多出了一个参数），其流程如下图所示：

![](.\img\Scaled Dot-Product Attention.jpg)

假设输入的query 维度为![[公式]](https://www.zhihu.com/equation?tex=q+) 、key维度为 ![[公式]](https://www.zhihu.com/equation?tex=d_k) ,value维度为 ![[公式]](https://www.zhihu.com/equation?tex=d_v) , 那么就计算query和每个key的点乘操作，并除以 ![[公式]](https://www.zhihu.com/equation?tex=%5Csqrt%7Bd_k%7D) ，然后应用Softmax函数计算权重。在实际操作通常都是多个query同时进行计算，query组成一个矩阵Q，key和value也组成矩阵K和V，因此矩阵输出如下：

![](.\img\Scaled Dot-Product Attention计算公式.jpg)

这样不是很形象，可以结合下文的的[Self Attetnion微观角度](#####微观角度)来理解。

但是如果只对Q、K、V做一次这样的权重操作是不够的，因此原文又提出了Multi-Head Attention，其流程如下：

<img src=".\img\Multi-Head Attention.jpg" style="zoom:70%;" />

Query，Key，Value首先进过一个线性变换，然后输入到Scaled Dot-Product Attention进行计算 ，注意这里要做h次，其实也就是所谓的多头，每一次算一个头，而且每次Q，K，V进行线性变换的参数W是不一样的。计算结束之后将这h个结果拼接起来，在做一个线性变换得到最后的输出。

关于Multi-head attention的公式，原文中的表示如下：

<img src=".\img\Multi-head attention 公式.jpg" style="zoom:70%;" />

##### 前馈网络

在进行了Attention操作之后，encoder和decoder中的每一层都包含了一个全连接前向网络，对每个position的向量分别进行相同的操作，包括两个线性变换和一个ReLU激活输出。

<img src=".\img\Position-Wise 全连接层公式.jpg" style="zoom: 50%;" />

其中每一层的参数都是不相同的。

##### Posistion Embedding

关于embedding通俗的翻译可以认为是单词嵌入，就是把X所属空间的单词映射为到Y空间的多维向量，那么该多维向量相当于嵌入到Y所属空间中，一个萝卜一个坑。

因为模型不包括recurrence/convolution，因此是无法捕捉到序列顺序信息的，例如将K、V按行进行打乱，那么Attention之后的结果是一样的。但是序列信息非常重要，代表着全局的结构，因此必须将序列的token相对或者绝对position信息利用起来。

这里每个token的position embedding 向量维度也是 ![[公式]](https://www.zhihu.com/equation?tex=d_%7Bmodel%7D%3D512%2C) 然后将原本的input embedding和position embedding加起来组成最终的embedding作为encoder/decoder的输入。原文中使用的计算公式如下：

<img src=".\img\position embedding计算公式.jpg" style="zoom:50%;" />

#### 1.3 Self Attention

##### 宏观角度

Self Attention 是Transformer的一个核心内容，他其实就是自己和自己的Attention机制。self attention会给你一个矩阵，告诉你 entity1 和entity2、entity3 ….的关联程度、entity2和entity1、entity3…的关联程度。它指的不是Target和Source之间的Attention机制，而是Source内部元素之间或者Target内部元素之间发生的Attention机制，也可以理解为Target=Source这种特殊情况下的注意力计算机制，也就是Q=K=V。**随着模型处理输入序列的每个单词，自注意力会关注整个输入序列的所有单词，帮助模型对本单词更好地进行编码**。

##### 微观角度

计算自注意力的第一步就是从每个编码器的输入向量（每个单词的词向量）中生成三个向量。也就是说对于每个单词，我们创造**一个查询向量、一个键向量和一个值向量**。这三个向量是通过词嵌入与三个权重矩阵后相乘创建的。这些新向量在维度上比词嵌入向量更低，他们的维度是64，而词嵌入和编码器的输入/输出向量的维度是512. 但实际上不强求维度更小，这只是一种基于架构上的选择，它可以使多头注意力（multiheaded attention）的大部分计算保持不变。

计算自注意力的第二步是计算得分。假设我们在为一个例子中的第一个词“Thinking”计算自注意力向量，我们需要拿输入句子中的每个单词（包括他自己）对“Thinking”打分。这些分数决定了在编码单词“Thinking”的过程中有多重视句子的其它部分。这些分数是通过打分单词（所有输入句子的单词）的键向量与“Thinking”的查询向量相点积来计算的。

第三步和第四步是将分数除以8(8是论文中使用的键向量的维数64的平方根，这会让梯度更稳定。这里也可以使用其它值，8只是默认值)，然后通过softmax传递结果。softmax的作用是使所有单词的分数归一化，得到的分数都是正值且和为1。这二步和第三步合起来的打分就对应上文[Attention Machanism](#####Attention Machanism)中的计算权重的步骤。

第五步是将每个值向量乘以softmax分数，然后将这些甲醛向量求和。这里的直觉是希望关注语义上相关的单词，并弱化不相关的单词(例如，让它们乘以0.001这样的小数)，然后即得到自注意力层在该位置的输出。接下来就是将这个输出传到前馈网络中的，这就和自注意力机制无关了。

#### 1.4 Transformer梳理

首先从宏观来看，将Transformer模型视为一个黑箱操作，原文的Transformer主要是应用在机器翻译上的，因此它的输入是一种语言输出是另一种语言。拆开这个黑箱就是编码器和解码器，编码器模块和解码器模块内都有多层。详见[Transfomer架构](###1.2 Transformer 架构)中的架构图。

模型结构很简单，重点是张量怎样在模型中一步步由输入转换为输出：

1. 首先（还是放在NLP中考虑）将每个输入单词通过嵌入算法（embedding）转换为词向量，原文中是将每个单词嵌入512维的词向量，这个过程发生在进入编码器之前。
2. 接下来就是进入编码器，所有的编码器都有一个相同的特点，即它们接收一个向量列表，列表中的每个向量大小为512维。在底层（最开始）编码器中它就是词向量，但是在其他编码器中，它就是下一层编码器的输出（也是一个向量列表）。向量列表大小是我们可以设置的超参数——一般是我们训练集中最长句子的长度。关于编码器内部进行的操作上文已经阐述清楚了。
3. 编码器通过处理输入序列开启工作。顶端编码器的输出之后会变转化为一个包含向量K（键向量）和V（值向量）的注意力向量集 。这些向量将被每个解码器用于自身的“**编码-解码注意力层**”，而这些层可以帮助解码器关注输入序列哪些位置合适。解码阶段的每个步骤都会输出一个输出序列的元素。接下来的步骤重复了这个过程，直到到达一个特殊的终止符号，它表示transformer的解码器已经完成了它的输出。每个步骤的输出在下一个时间步被提供给底端解码器，并且就像编码器之前做的那样，这些解码器会输出它们的解码结果 。另外，就像我们对编码器的输入所做的那样，我们会嵌入并添加位置编码给那些解码器，来表示每个单词的位置。除此之外，在解码器中自注意力层只被允许处理输出序列中更靠前的那些位置。在softmax步骤前，它会把后面的位置给隐去（把它们设为-inf）

### 二、图片分类部分

论文链接：[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

#### 2.1 简介

Transformer模型提出之后很快就在NLP领域占据了主导地位，但是在CV领域内CNN依然是主导，之前有过很多尝试将Transformer和CNN结合或者是完全使用Transformer替代CNN，但是后者并没有成功实践过。原文实验了在对Transformer最小改动的情况下直接应用在图片上。为此作者将一个图像分成很多片，然后将这些分片的线性embedding作为Transformer的输入，即将这些分片想单词一样处理。这么做的话Transformer在小或中等大小的数据集上训练时效果不佳（因为Transformer相较CNNs，缺少一些固有的归纳偏置，例如：平移不变性和空间局部性），但是如果使用很大的数据集（比ImageNet数据集还要大的数据集）训练Transformer的效果非常好。

#### 2.2 具体实现

##### 整体架构

Vision Transformer（ViT）的整体架构如下图所示：

<img src=".\img\图像分类Transformer模型架构.jpg" style="zoom:80%;" />

标准的接收的是一维的token embeddings序列，为了检测二维的图像，作者将每一张输入图片（H x W x C），切割为（P × P × C）的小块，并将小块flatten为P × P × C的单维向量，最后得到（N，P×P×C）的序列。因为Transformer的输入维度为固定大小D，因此，我们用一个trainable linear projection将patches投影为D维。同时作者还预置了一个可学习的embedded patches xclass，作为图片的标签输出。在pre-training和fine-tuning过程中，xclass始终作为classification head的输入。而关于位置信息，作者使用了标准的可学习一维嵌入（实验表明使用二维位置嵌入对结果并没有什么改善）。

Transformer Encoder的结构如上图右侧所示，了解了标准Transformer以后别这个就很好理解。

输入除了可以是图像的切片转换而来也可以是从特征图上产生，这种被称为混合模型。

##### fine-tune和高分辨率

典型的方式是首先使用大数据集与训练ViT，然后将预训练好的预测头换成由0初始化的D×K的前馈层（D是输入维度、K是下游分类任务的类别数量）。

当输入图片分辨率变高时，保持分片大小不变，使输入序列变长，ViT可以处理任意长度的输入序列，但是会导致pre-trained position embeddings失去意义，因此，作者根据它的位置，对它进行二维插值。注意，这里的resolution adjustment and patch extraction是图片2D结构唯一的归纳偏置，被我们人为加入到ViT当中

### 三、识别部分

上一节提到的再图像中使用Transformer只包括图像分类，而目前Transformer模型已经应用到其他方面了（比如目标检测），论文链接：[End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

#### 3.1 简介

原文介绍的Detection Transformer（DETR）是第一个将 Transformer 成功整合为检测 pipeline 中心构建块的目标检测框架。基于Transformers的端到端目标检测，没有NMS后处理步骤、真正的没有anchor，且对标超越Faster RCNN。

目标检测的目标是预测一个bbox的集合和各个bbox的标签。目前的检测器不是直接预测一个目标的集合，而是使用替代的回归和分类去处理大量的propoasls、anchors或者window centers。为了简化流程，作者提出一种直接set prediction的方式来消除这些替代的方法：原文**把目标检测当做一种集合预测的问题来处理**，采用了一种常用的序列预测架构，使用编码-解码器的transformer。这种架构可以显示的对序列中元素的两两关系来建模，容易满足一些集合预测的限制，比如消除冗余。我们把目标检测当做一种集合预测的问题来处理。我们采用了一种常用的序列预测架构，使用编码-解码器的transformer。具体如下图所示：

![](.\img\Transformer目标检测架构.jpg)

DETR一次性预测所有目标对象，成对的匹配预测目标与真实目标并使用固定的损失函数进行端对端训练。DETR移除了大量人工设计的模块，比如空间上的anchors和NMS。同时它不需要任何定制的网络层，因此可以在任何支持CNN和transformer的框架下复现。DETR有两个主要的特征，**两两匹配的loss和并行解码的transformers**。实验表明DETR在大型目标识别上的表现比最新的Faster R-CNN还要好，不过在小目标上就差一些了，同时DETR还可以应用许多其他任务上去。

##### 集合预测

之前提到过DETR将目标检测作为一种集合预测问题来处理，常见的方法应该是设计一种基于匈牙利算法的损失，来找到真实值与预测值的二分匹配。这种可以保证损失不受排列的影像而且保证每一个目标有唯一一个匹配。我们依照着二分匹配损失的方法。不同于之前的其他工作，我们不使用自回归的模型，转而使用并行解码的transformer。

#### 3.2 DETR模型架构

目标检测中使用直接集合预测最关键的两个点是：1）保证真实值与预测值之间**唯一匹配的集合预测损失**。2）一个可以预测（一次性）目标集合和对他们关系建模的架构。

##### 目标检测集合预测损失

DETR每次预测预测固定的N个预测值，N是提前设定的并且显著大于图像中含有的目标数量。有一个难点是如何为这些预测打分（包括分类分数、位置分数、大小分数）并进行bbox回归。因此作者设计的损失构造了一个最优的二分匹配而且接着优化目标向（bounding box）的损失。

我们将y指示真实值， ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D%3D%5C%7B%5Chat%7By_i%7D%5C%7D%5E%7BN%7D_%7Bi%3D1%7D) 指示N个预测值。假设N远大于图像中的目标，我们可以认为y的大小也是N，用 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi) （空对象）填充空元素。目标就是找到这两个集合的二分匹配，中的一种排列 ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma) 有着最低的损失：

<img src=".\img\DETR配对损失计算.jpg" style="zoom:80%;" />

上述loss是真实值与预测值之间两两匹配的loss，使用匈牙利算法来计算。这个配对损失包含分类损失和定位损失，大概的思路就是遍历所有的一一配对找到总损失值最低的一种组合方式。寻找匹配的方式和最新的检测器匹配proposal或者anchor的方式相似，是**启发式的搜索 heuristic search**（详见[启发式搜索](https://www.cnblogs.com/ISGuXing/p/9800490.html)），只不过是一一配对没有冗余，要注意的是匹配时所用的损失和下一步计算最终损失时的损失函数不一样，匹配时直接使用概率而下一步计算时加了负对数操作，作者给出的理由是这样效果更好。第二步是计算损失函数，关于损失函数原文的定义如下：

<img src=".\img\DETR损失函数.jpg" style="zoom:67%;" />

其实就是负对数似然预bbox损失的线性组合，在实践中，类似于faster-rcnn对负样本权重的设置，当 ![[公式]](https://www.zhihu.com/equation?tex=c_i+%3D+%5Cphi) 时，权重为原来的十分之一。目标与 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi) 的匹配损失不依赖于预测值，因此是一个常量。而关于bbox 损失作者使用了L1 loss 与IOU loss的组合让loss对目标的大小不敏感，同时每一个batch内部使用目标数量对loss进行了归一化。

##### 整体架构

<img src=".\img\DETR整体架构.jpg" style="zoom:90%;" />

如上图所示，DETR主要包含三个部分：一个提取图像特征的CNN，一个编码-解码的transformer，一个用来预测最终目标的前向网络FFN。

###### backbone

输入图像尺寸为 H0 × W0 × 3，最终生成一个2048 × H0/32 × W0/32 的低分辨率特征图（不是一定是这个尺寸，不过一般典型的都是这个尺寸）。原文这里使用了activation map而不是feature map，不过应该没啥区别。

###### encoder & encoder

首先，一个1×1的卷积层把之前C个通道（就是那个典型为2048）的特征图通道数转化为d，然后因为encoder期望输入是一个序列，还要将新的 d × W × H 的特征图压成 d × WH。之后就是标准的Transformer结构，由多层的multi-head self-attention module 和 feed forward network (FFN)组成。由于transformer对排列顺序不敏感，所以作者还加入了位置的编码，并添加到所有attention层的输入。

DETR Decoder的结构也与Transformer类似，每个Decoder有两个输入：一个是Object Query（或者是上一个Decoder的输出），另一个是Encoder的结果，区别在于这里是并行解码N个object。与原始的transformer不同的地方在于decoder每一层都输出结果，计算loss。另外一个与Transformer不同的地方是，DETR的Decoder和encoder一样也加入了可学习的positional embedding，其功能类似于anchor。最后一个Decoder后面接了两个FFN，分别预测检测框及其类别。

*关于这个object queries原文的解释是a small fixed number of learned positional embeddings*

###### FFNs

最后的预测是由一个三层感知机（包括ReLU激活函数、维度为d的隐藏层）和一个线性投影层产生的。FFN产生box中心坐标和长宽，而线性层使用softmax函数产生分类预测。

###### 附加的解码loss

为了更好的训练模型，作者还在每一个decoder层后面加上FFNs和匈牙利loss。所有的预测FFNs使用共享的参数，并使用额外的共享层来归一化不同decoder的输出。

#### 3.3 应用

DETR可以很好地应用在 Panoptic  segmentation 全景分割上。作者在每个预测的box上加了一个预测二值掩码的mask head（拓展思路和Faster R-CNN到Mask R-CNN类似），这些mask接受decoder的输出作为输入然后为每个对象生成M（M为Multi-head的head数）个attention heatmaps，之后还使用了一个FPN-like的结构。而为了最终得到全局分割结果就简单地在每一个像素上应用一个选择机制，依据mask评分分配相应的类。



































