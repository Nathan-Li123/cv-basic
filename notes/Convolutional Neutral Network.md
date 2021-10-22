# Convolution Neutral Network

### 一、神经网络

在学习卷积神经网络之前先了解一下神经网络。

#### 1.1 神经网络分析

<img src=".\img\两层神经网络.jpg" alt="两层神经网络" style="zoom: 80%;" />

##### 输入层

在例子中输入值是坐标值，是一个包含两个元素的一维数组。可以理解为输入层为输入一个矩阵，图中的就是1×2矩阵，输入一张32×32像素的灰色图像就是输入32×32的矩阵。

##### 从输入层到隐藏层

从输入层到隐藏层依靠权重矩阵W1和偏置b1，其实就是进行矩阵运算：H = X × W1 + b1。包括隐藏层之间传输和从隐藏层到输出层也是一样的计算方法，只不过用了不同的权重矩阵和偏置。

##### 激活层

激活层需要在每个隐藏层计算之后，不然该层的线性计算就是没有意义的。激活层，或是激励层在之后卷积神经网络中展开。

##### 输出层

数据到达输出层之后其实我们已经拿到了一个结果，比如图中在softmax之前输出的值可能会是(3,1,0.1,0.5)这样的1×4矩阵，我们已经可以找打这里面的最大值3了，但是这个结果不够直观，最好是直接输出属于某个类的概率，比如输出(90%,5%,2%,3%)这样的结果，因此我们还需要进行输出正规化。

简单来说分为三步：（1）以e为底对所有元素求指数幂；（2）将所有指数幂求和；（3）分别将这些指数幂与该和做商。其实就是每一项指数幂求和，然后概率就是单项指数幂占总数的比例。使用这种计算公式进行结果正规化处理就称为Softmax。

#### 1.2 衡量输出好坏

经过Softmax之后就得到了一个结果，这时候需要对这个输出结果的好坏程度进行一个“量化”，这时候就需要一个损失函数。

##### 交叉熵损失

比较典型的一种量化方法是使用**交叉熵损失( Cross Entropy Loss )**。这个损失的想法很简单，就是求对数的负数，比如对于0.9，结果就是-log0.9 = 0.046，概率越接近100%，交叉熵损失就越接近0。交叉熵损失也被称为log对数损失函数。

##### Hinge损失

Hinge损失函数标准形式如下：

<img src="https://www.zhihu.com/equation?tex=L%28y%2C+f%28x%29%29+%3D+max%280%2C+1-yf%28x%29%29+++%5C%5C" alt="[公式]" style="zoom:80%;" />

hinge损失函数表示如果被分类正确，损失为0，否则损失就为 ![[公式]](https://www.zhihu.com/equation?tex=1-yf%28x%29) 。SVM分类器就是使用这个损失函数，健壮性不错但是对异常点和噪音不太敏感。hinge损失函数还有一种变体称为感知损失函数。在感知损失函数中，分类正确时损失为，分类不正确损失为负的预测值，他对判定边界附近的点非常严格。

##### Smooth L1 损失

smooth L1说的是光滑之后的L1（绝对值平均损失），L1损失的缺点就是有折点，不光滑，导致不稳定。smooth L1损失函数为： 
                                                               ![smooth L1](https://img-blog.csdnimg.cn/20190621105252220.png)

还有一些损失函数，例如均方差损失函数、0-1损失函数、绝对值损失函数、指数损失函数等

#### 1.3 反向传播

##### 反向传播实现

之前说过神经网络的传播都是形如 Y=WX+b 的矩阵运算，为了给矩阵运算加入非线性，需要在隐藏层中加入激活层。在计算出交叉熵损失之后就要反向传播来进行参数优化，使结果最为准确，这里的参数指的就是W和b，毕竟只有这两个是可变的。反向传播的原理很简单，就是不断地微调这些可变参数，然后再次计算损失值，不断迭代，使损失值越来越小，直到我们满意为止。

在正向传播的时候一个矩阵经过权重矩阵运算得到一个结果，然后在经过激活函数得到该层的最终输出，因此整个变换过程可以视为一个多元函数。考虑如下图所示的神经网络：

<img src=".\img\神经网络02.jpg" style="zoom:50%;" />

图中h1的输出值就是以w1, w2和b1为参数的一个函数再经过激励函数得到的，同理h1输出值的loss也是由这些参数得到，因此在反向传播的时候就可以分别求出loss值关于参数w1、w2和b1的梯度，然后根据梯度下降算法更新这些参数的值，并且按照这种方式不断向输入层传播，直到全部的参数都被更新了。一遍更新完成之后就再次计算总的loss，然后再次迭代反向传播，直到loss令人满意为止。

##### 梯度下降

梯度下降算法背后的原理：目标函数 ![[公式]](https://www.zhihu.com/equation?tex=J%28%5Ctheta%29) 关于参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) 的梯度将是损失函数（loss function）上升最快的方向。而我们要最小化loss，只需要将参数沿着梯度相反的方向前进一个步长，就可以实现目标函数（loss function）的下降。这个步长 ![[公式]](https://www.zhihu.com/equation?tex=%5Ceta) 又称为学习速率。也就是说最基础的参数的更新公式为：

![公式](https://www.zhihu.com/equation?tex=%5Ctheta%5Cleftarrow+%5Ctheta-%5Ceta+%5Ccdot+%5Cnabla+J%28%5Ctheta%29)

其中 ![[公式]](https://www.zhihu.com/equation?tex=+%5Cnabla+J%28%5Ctheta%29) 是参数的梯度。梯度下降算法可以分为批量梯度下降算法（Batch Gradient Descent），随机梯度下降算法（Stochastic Gradient Descent)和小批量梯度下降算法（Mini-batch Gradient Descent)。

* 批量梯度下降算法(BGD)：目标函数是在整个训练集上计算的，如果数据集很大就会很慢，甚至内存不足。
* 随机梯度下降算法(SGD)：和批量完全相反，只在一个样本上计算。
* 小批量梯度下降算法(MBGD)：算是一个折中方案，选用数据集的一部分进行计算。

##### 改进的梯度下降算法

###### 冲量梯度下降

冲量梯度下降算法（Momentum optimization），他的思路形象表示是这样的：将一个小球从山顶滚下，其初始速率很慢，但在加速度作用下速率很快增加，并最终由于阻力的存在达到一个稳定速率。冲量梯度下降的表达式如下：

![[公式]](https://www.zhihu.com/equation?tex=m%5Cleftarrow+%5Cbeta+%5Ccdot+m%2B%281-%5Cbeta%29%5Ccdot%5Cnabla+J%28%5Ctheta%29)

![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta+%5Cleftarrow+%5Ctheta+-+%5Ceta+%5Ccdot+m)

可以看到，参数更新时不仅考虑当前梯度值，而且加上了一个积累项（冲量），但多了一个参数伽马，一般取接近1的值如0.9。相比原始梯度下降算法，冲量梯度下降算法有助于加速收敛。形象来说这种梯度下降有可能让小球冲出不是最低点的凹槽。

###### Nesterov Accelerated Gradient (NAG)

该梯度下降算法的变化之处在于计算“超前梯度”更新冲量项，也就是说他不只考虑当前的梯度还考虑下一节点的梯度，具体公式如下：

![[公式]](https://www.zhihu.com/equation?tex=m%5Cleftarrow+%5Cgamma+%5Ccdot+m%2B%5Ceta+%5Ccdot%5Cnabla+J%28%5Ctheta+-%5Cgamma+%5Ccdot+m+%29)

![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta+%5Cleftarrow+%5Ctheta+-+m)

该公式考虑了沿当前冲量往下走的下一项的梯度，合并两项作为最终的更新项。

###### AdaGrad

AdaGrad是Duchi在2011年提出的一种学习速率自适应的梯度下降算法。在训练迭代过程，其学习速率是逐渐衰减的，经常更新的参数其学习速率衰减更快，这是一种自适应算法。 其更新过程如下：

![[公式]](https://www.zhihu.com/equation?tex=s%5Cleftarrow+s+%2B+%5Cnabla+J%28%5Ctheta%29%5Codot+%5Cnabla+J%28%5Ctheta%29)

![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta+%5Cleftarrow+%5Ctheta-%5Cfrac%7B%5Ceta%7D%7B%5Csqrt%7Bs%2B%5Cvarepsilon%7D%7D%5Codot+%5Cnabla+J%28%5Ctheta%29)

其中是梯度平方的积累量 ![[公式]](https://www.zhihu.com/equation?tex=s) ，在进行参数更新时，学习速率要除以这个积累量的平方根，其中加上一个很小值 ![[公式]](https://www.zhihu.com/equation?tex=%5Cvarepsilon) 是为了防止除0的出现。这种方法的主要改进是加速收敛，当接近坡底积累量变小，但积累量是作为除数所以能保证收敛速度还是足够快。问题是因为学习速率是不断缩减的，所以可能导致学习过早地停止。

###### RMSprop

这是对AdaGrad的缺陷改进的梯度下降算法，其实思路很简单，类似Momentum思想，引入一个超参数，在积累梯度平方项进行衰减：

![[公式]](https://www.zhihu.com/equation?tex=s%5Cleftarrow+%5Cgamma%5Ccdot+s+%2B+%281-%5Cgamma%29%5Ccdot+%5Cnabla+J%28%5Ctheta%29%5Codot+%5Cnabla+J%28%5Ctheta%29)

![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta+%5Cleftarrow+%5Ctheta-%5Cfrac%7B%5Ceta%7D%7B%5Csqrt%7Bs%2B%5Cvarepsilon%7D%7D%5Codot+%5Cnabla+J%28%5Ctheta%29)

其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma) 一般取值0.9，此时 ![[公式]](https://www.zhihu.com/equation?tex=s) 更平稳，减少了出现的爆炸情况，因此有助于避免学习速率很快下降的问题。

###### Adam

Adaptive moment estimation (Adam)算法类似于Momentum和RMSprop的结合：

![[公式]](https://www.zhihu.com/equation?tex=m%5Cleftarrow+%5Cbeta+_%7B1%7D+%5Ccdot+m%2B%281-%5Cbeta+_%7B1%7D%29%5Ccdot%5Cnabla+J%28%5Ctheta%29)

![[公式]](https://www.zhihu.com/equation?tex=s%5Cleftarrow%5Cbeta+_%7B2%7D%5Ccdot+s+%2B+%281-%5Cbeta+_%7B2%7D%29%5Ccdot+%5Cnabla+J%28%5Ctheta%29%5Codot+%5Cnabla+J%28%5Ctheta%29)

![[公式]](https://www.zhihu.com/equation?tex=m%5Cleftarrow+%5Cfrac%7Bm%7D%7B1-%5Cbeta+_%7B1%7D%5E%7Bt%7D%7D)

![[公式]](https://www.zhihu.com/equation?tex=s%5Cleftarrow+%5Cfrac%7Bs%7D%7B1-%5Cbeta+_%7B2%7D%5Et%7D)

![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta+%5Cleftarrow+%5Ctheta-%5Cfrac%7B%5Ceta%7D%7B%5Csqrt%7Bs%2B%5Cvarepsilon%7D%7D%5Codot+m)

### 二、卷积神经层级架构

#### 2.1 数据输入层

数据输入层主要做的是对原始数据进行预处理。

* 去均值：把输入数据各个维度都中心化为0，其目的就是把样本的中心拉回到坐标系原点上。
* 归一化：幅度归一化到同样的范围，即减少各维度数据取值范围的差异而带来的干扰，一般归一至0到1的范围
* PCA/白化：PCA降维；白化是对数据各个特征轴上的幅度归一化

#### 2.2 卷积计算层

卷积计算层是卷积神经网络最重要的一个层次，有两个关键操作：**局部关联**和**窗口滑动**

* 深度：每个神经元看作一个滤波器 filter，有多少个神经元，深度depth就是多少
* 步长：步长stride是窗口滑动一次的长度
* 填充值：有些时候滑动窗口无法遍历所有的像素，这时候就需要在原先矩阵填充几层，填进去的就是填充值padding，通常使用zero-padding也就是使用0来填充

##### 卷积的计算

卷积操作中，一个 3×3×3 的子节点矩阵和一个 3×3×3 的 filter 对应元素相乘，得到的是一个 3×3×3 的矩阵，此时将该矩阵所有元素求和，得到一个 1×1×1 的矩阵，将其再加上 filter 的 bias，经过激活函数得到最后的结果，将最后的结果填入到对应的输出矩阵中。

<img src=".\img\卷积计算01.jpg" alt="卷积计算01" style="zoom:80%;" />

这里的蓝色矩阵就是输入的图像，粉色矩阵就是卷积层的神经元，这里表示了有两个神经元（w0,w1）。绿色矩阵就是经过卷积运算后的输出矩阵，这里的步长设置为2。该实例中输入矩阵为7×7×3，滤波器为3×3×3，而输出矩阵为3×3×2，滑动窗口每移动一个步长，**计算三层的窗口矩阵与滤波器矩阵对应位相乘再相加，最后加上偏置值得到一个输出**（例如图中就是 0 + (-2-2) + 0 + 1 = -3 ）的计算，最终得到3×3的矩阵，因为有两个滤波器所以输出矩阵层数为2。

<img src="E:\李云昊\国科\computer-vision\notes\img\单层卷积网络.jpg" alt="单层卷积网络"  />

##### 参数共享机制

在卷积层中每个神经元连接数据窗的权重是固定的，每个神经元只关注一个特性。神经元就是图像处理中的滤波器，比如边缘检测专用的Sobel滤波器，即卷积层的每个滤波器都会有自己所关注一个图像特征，比如垂直边缘，水平边缘，颜色，纹理等等，这些所有神经元加起来就好比就是整张图像的特征提取器集合。

#### 2.3 ReLU激励层

把卷积层输出结果做非线性映射。CNN采用的激励函数(激活函数)一般是ReLU( The Rectified Linear Unit 修正线性单元 )，它的特点是收敛快，求梯度简单，但是比较脆弱。

神经网络中每一层进行矩阵运算之后得到结果，激活函数或者说是激励函数就是为了给该结果添加非线性用的，常见的激活函数有阶跃函数、Sigmoid函数和ReLU函数，基础CNN中通常使用ReLU。三种函数见下图所示：

<img src=".\img\激活函数.jpg" alt="三种激活函数" style="zoom:80%;" />

##### ReLU

ReLU及其改进型是近年来用的比较多的激励函数。标准ReLU函数很简单，就是小于0时为0，大于0时不变。

###### Leaky ReLU

Leaky ReLU，其表达式为                                <img src="C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20211019192649835.png" alt="image-20211019192649835" style="zoom:80%;" />

理论上来讲，Leaky ReLU有ReLU的所有优点，外加不会有Dead ReLU问题，但是在实际操作当中，并没有完全证明Leaky ReLU总是好于ReLU。

PReLU和RReLU表达式和Leaky ReLU一样的，可以视为Leaky ReLU的变体，区别是PReLU的参数 α 不是固定的，而是根据数据来决定，而RReLU的区别是在训练环节中，α 是从一个均匀的分布U(I,u)中随机抽取的数值。

###### Exponential Linear Units

函数表达式：                                                    <img src="E:\李云昊\国科\computer-vision\notes\img\ELU表达式.jpg" style="zoom:80%;" />    

ELU也是为解决ReLU存在的问题而提出，他不会有Dead ReLU问题，同时输出的均值也接近0，但是和Leaky ReLU一样，实际运用中并没有证明其比ReLU有效。

##### 必要性

线性代数有一个特性，**一系列线性方程的运算最终都可以用一个线性方程表示**。所以无论神经网络有多少层，多个式子联立之后都可以用一个线性方程表达，这样的话神经网络就失去了意义，因此需要激活层（ReLU比较脆弱的意思就是比较容易被破解）。

#### 2.4 池化层

池化层夹在连续的卷积层中间， 用于压缩数据和参数的量，减小过拟合。简而言之，**如果输入是图像的话，那么池化层的最主要作用就是压缩图像**，去除无用信息。

*特征不变性：图像压缩时去掉的信息只是一些无关紧要的信息，而留下的信息则是具有尺度不变性的特征，是最能表达图像的特征。*

##### 具体作用

* 特征降维：一幅图像含有的信息是很大的，特征也很多，但是有些信息对于我们做图像任务时没有太多用途或者有重复，我们可以把这类冗余信息去除，把最重要的特征抽取出来。
* 在一定程度上防止过拟合，更方便优化。
* 池化能够有效增大感受野。

##### 池化层方法

池化层一般不改变矩阵深度，只改变长和宽。池化层常方法有 Max Pooling 和 Average Pooling，实际使用比较多的是Max Pooling。

<img src=".\img\max pooling.jpg" alt="Max Pooling" style="zoom:67%;" />

对于每个2 * 2的窗口选出最大的数作为输出矩阵的相应元素的值，比如输入矩阵第一个2 * 2窗口中最大的数是6，那么输出矩阵的第一个元素就是6，如此类推。同样的 Average Pooling 就是去平均值，不再展开。

除此之外还有一些其他的方法：

* stochastic pooling：随机池化，对feature map中的元素按照其概率值大小随机选择，元素被选中的概率与其数值大小正相关。
* mixed pooling：在max/average pooling中进行随机选择。

#### 2.5 全连接层

CNN中两层之间所有神经元都有权重连接，通常全连接层在卷积神经网络尾部，也就是跟传统的神经网络神经元的连接方式是一样的。经过多轮卷积层和池化层的处理后，在CNN的最后一般由1到2个全连接层来给出最后的分类结果。经过几轮卷积和池化操作，可以认为图像中的信息已经被抽象成了信息含量更高的特征。我们可以将卷积和池化看成自动图像提取的过程，在特征提取完成后，仍然**需要使用全连接层来完成分类任务**。

### 三、实例记录

实例使用了一个卷积神经网络对MNIST数据集进行分类，代码位于 。主要在此记录一下编写调试过程中遇到的问题和重点。

##### 卷积时Tensor变化

首先输入Tensor为（8，1，28，28），这代表batch_size为8，图像channel为1，分辨率为28×28。之后经过第一个5×5的卷积核，stride和padding都为默认值，即1和0，可以看到Tensor变为了（8，6，12，12）。

<img src=".\img\卷积中张量变化01.jpg" style="zoom: 80%;" />

过程：因为padding为0所有外围没有填充，那么卷积计算过后张量特征矩阵就变成了24×24，ReLU激励函数不改变矩阵shape，最后经过2×2的最大池，矩阵变成了12×12.同理第二次经过Conv2d(6, 6, 5)的池子之后变为（8，6，4，4）。

至此卷积计算完成进入神经网络，进入之前我们将四维张量reshape成二维的。个人理解reshape过后应当令第一维的shape是不变的，也就是让二维矩阵的行数等于batch_size，每行对应一个传入的图像的特征向量。

<img src=".\img\卷积中张量变化02.jpg" style="zoom:80%;" />

从结果中可以看到确实是变换成了8×96，但是再view的时候使用的是列数96然后自动计算行数，实际上是需要自己算好的得到96。之后的过程中行数不再发生变换，列数最终是变成10，这也是设计卷积神经网络时确定的最终分类的类别数量。整个model最后得到的是（8，10）的Tensor，这和训练时传入的labels是一样的shape。

***所以卷积计算和神经网络Linear的shape设置是要算过的，别随便设！***

##### 输出分类结果时技巧

```python
_, prediction = torch.max(output.data, 1)
correct += (prediction == target).sum().item()
c = (prediction == target).squeeze()
for i in range(8):
    label = target[i]
    class_correct[label] += c[i].item()
    class_total[label] += 1
```

来自PyTorch官方教程的输出方式，学习一下，还真没见过 (prediction == target).squeeze() 这种操作。

### 四、Notes

##### channel是什么？

* 最初输入的图片样本的 channels ，取决于图片类型，比如RGB类型就是3。
* 卷积操作完成后输出的 out_channels ，取决于卷积核的数量。此时的 out_channels 也会作为下一次卷积时的卷积核的 in_channels。
* 卷积核中的 in_channels ，刚刚2中已经说了，就是上一次卷积的 out_channels ，如果是第一次做卷积，就是1中样本图片的 channels 。

##### 神经网络与卷积神经网络

从广义上来说卷积神经网络也属于多级神经网络的一种，他在原来多级神经网络的基础上加入了特征学习的部分。具体操作就是在原来的全连接层钱买你加入了部分连接的卷积层和激活层，使完整的层级变为：**输入层 - 卷积层 - 激活层 - …… - 隐藏层 - 输出层**。或者可以直接理解为卷积神经网络是在神经网络分类之前先利用卷积进行特征提取，卷积不是结合在神经网络里面的！**~~不要理解为神经网络的隐藏层中进行卷积运算！~~**

### 五、Reference

* [神经网络15分钟入门！足够通俗易懂了吧](https://zhuanlan.zhihu.com/p/65472471)
* [卷积神经网络(CNN)  手记](https://www.imooc.com/article/68983)
* [吴恩达 卷积神经网络 CNN  简书](https://www.jianshu.com/p/da0db799a681)
* [神经网络反向传播算法  知乎](https://zhuanlan.zhihu.com/p/25609953)
* [一文看懂常用的梯度下降算法  知乎](https://zhuanlan.zhihu.com/p/31630368)
