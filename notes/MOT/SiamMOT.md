# SiamMOT

论文链接：[SiamMOT: Siamese Multi-Object Tracking](https://arxiv.org/abs/2105.11595)

### 一、简介

概述：通过引入一个基于区域的孪生多目标跟踪网络，作者设计了一个新的online多目标跟踪框架，名为SiamMOT。SiamMOT包含一个运动模型来估计两帧之间目标的移动从而关联两帧上检测到的目标。

作者基于SORT（Simple and On-line Realtime Tracking，这是很多最先进的模型的基础），在SORT中，一个更好的**运动模型**是提高跟踪精度的关键。作者利用基于区域的孪生多目标跟踪网络来进行运动建模的探索，称其为SiamMOT。作者组合了一个基于区域的检测网络（Faster R-CNN）和两个思路源于孪生单目标跟踪的运动模型（分别是隐式运动模型（IMM）和显式运动模型（EMM））。不同于CenterTrack基于点的特征进行隐式的目标运动预测，SiamMOT使用基于区域的特征并且开发了显式的模板匹配策略来估计模板的运动.

### 二、SiamMOT 实现

SiamMOT是基于Faster R-CNN构建的，其整体架构如下图所示：

<img src="..\img\SiamMOT架构.jpg" style="zoom:80%;" />

如上图所示，SiamMOT再Faster R-CNN上又加了一个region-based Siamese tracker来建模实例级别的运动。SiamMOT获取两帧的输入和一些在第一帧检测到的对象，然后检测网络输出第二帧中的检测对象，追踪网络输出由第一帧检测框得到的第二帧的预测框，然后将两个输出进行匹配，从而将其关联起来形成轨迹。

#### 2.1 动作预测

上文提到的SiamMOT使用了一个Siamese tracker来进行实例级别的动作预测。假设传入了一个![[公式]](https://www.zhihu.com/equation?tex=t)中检测到的实例 i，在孪生跟踪器根据其在![[公式]](https://www.zhihu.com/equation?tex=t)帧中的位置在![[公式]](https://www.zhihu.com/equation?tex=t%2B%5Cdelta)帧的一个局部窗口范围内搜索对应的实例，形式上表述如下：

![](..\img\Siamese Tracker 公式.jpg)

这里的![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BT%7D)表示参数为![[公式]](https://www.zhihu.com/equation?tex=%5CTheta)的孪生跟踪器，![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bf%7D_%7BR_%7Bi%7D%7D%5E%7Bt%7D)则是根据检测框![[公式]](https://www.zhihu.com/equation?tex=R_%7Bi%7D%5E%7Bt%7D)在![[公式]](https://www.zhihu.com/equation?tex=t)帧上提取的特征图，第二个参数则是在![[公式]](https://www.zhihu.com/equation?tex=t%2B%5Cdelta)帧上获得的搜索区域![[公式]](https://www.zhihu.com/equation?tex=S_%7Bi%7D%5E%7Bt%2B%5Cdelta%7D)提取的特征图。而搜索区域![[公式]](https://www.zhihu.com/equation?tex=S_%7Bi%7D%5E%7Bt%2B%5Cdelta%7D)的获得则通过按照比例因子![[公式]](https://www.zhihu.com/equation?tex=r)(r>1)来扩展检测框![[公式]](https://www.zhihu.com/equation?tex=R_%7Bi%7D%5E%7Bt%7D)来获得，拓展前后具有同样的集合中心（如上图中黄色放扩所示）。获得其特征![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bf%7D_%7BR_%7Bi%7D%7D%5E%7Bt%7D)和![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bf%7D_%7BS_%7Bi%7D%7D%5E%7Bt%2B%5Cdelta%7D)的方式都是不受大小影响的RoIAlign层。孪生跟踪器输出的结果有两个，其中![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7BR%7D_%7Bi%7D%5E%7Bt%2B%5Cdelta%7D)为预测框，而![[公式]](https://www.zhihu.com/equation?tex=v_%7Bi%7D%5E%7Bt%2B%5Cdelta%7D)则是预测框的可见置信度，若该实例在区域![[公式]](https://www.zhihu.com/equation?tex=S_%7Bi%7D%5E%7Bt%2B%5Cdelta%7D)是可见的，那么![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BT%7D)将会产生一个较高的置信度得分![[公式]](https://www.zhihu.com/equation?tex=v_%7Bi%7D%5E%7Bt%2B%5Cvarepsilon%7D)，否则得分会较低。这个公式会为每个实例执行一次，但是注意是可以并行计算的。

作者发现，运动建模对于多目标跟踪任务至关重要，一般在两种情况下![[公式]](https://www.zhihu.com/equation?tex=R%5E%7Bt%7D)和![[公式]](https://www.zhihu.com/equation?tex=R%5E%7Bt%2B%5Cdelta%7D)会关联失败，一是![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7BR%7D%5E%7Bt%2B%5Cdelta%7D)没有匹配上正确的![[公式]](https://www.zhihu.com/equation?tex=R%5E%7Bt%2B%5Cdelta%7D)，二是对于![[公式]](https://www.zhihu.com/equation?tex=t%2B%5Cdelta)帧上的行人得到的可见得分![[公式]](https://www.zhihu.com/equation?tex=v_%7Bi%7D%5E%7Bt%2B%5Cdelta%7D)太低了。对此，作者提出两种不同的孪生跟踪器：隐式运动模型和显式运动模型。

##### IMM（隐式）

它通过MLP来估计目标两帧间的运动。具体而言，它先将特征![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bf%7D_%7BR_%7Bi%7D%7D%5E%7Bt%7D)和![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bf%7D_%7BS_%7Bi%7D%7D%5E%7Bt%2B%5Cdelta%7D)按通道连接在一起然后送入MLP中预测可见置信度![[公式]](https://www.zhihu.com/equation?tex=v_i)和相对位置和尺度偏移，如下所示（相对位置和尺度偏移)：

<img src="..\img\IMM公式.jpg" style="zoom:90%;" />

训练时关于置信度使用focal loss，而关于预测框通常使用Smooth L1损失。

##### EMM（显式）

EMM整体流程如下图所示：

![](..\img\EMM架构.jpg)

具体而言，如上图所示，它通过逐通道的互相关操作（*）来生成像素级别的响应图，在SiamMOT中，这个操作用目标特征图![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bf%7D_%7BR_%7Bi%7D%7D%5E%7Bt%7D)和搜索图像特征图![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bf%7D_%7BS_%7Bi%7D%7D%5E%7Bt%2B%5Cdelta%7D)的每个位置计算相关性，得到![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Br%7D_%7Bi%7D%3D%5Cmathbf%7Bf%7D_%7BS_%7Bi%7D%7D%5E%7Bt%2B%5Cdelta%7D+%2A+%5Cmathbf%7Bf%7D_%7BR_%7Bi%7D%7D%5E%7Bt%7D)，所以每个![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Br%7D_%7Bi%7D%5Bk%2C%3A%2C%3A%5D)表示一个相似程度。受到FCOS的启发，EMM使用全卷积网络![[公式]](https://www.zhihu.com/equation?tex=%5Cpsi)来检测![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Br%7D_%7Bi%7D)中匹配的目标。

详细来看，![[公式]](https://www.zhihu.com/equation?tex=%5Cpsi)预测一个密集的**可见置信度图**![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bv%7D_%7Bi%7D)来表示每个像素包含目标图像的可能性，再预测一个**密集的定位图**![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bp%7D_%7Bi%7D)来编码该像素位置到框的左上角和右下角的偏移量。因此，处在![[公式]](https://www.zhihu.com/equation?tex=%28x%2Cy%29)的目标可以通过![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BR%7D%28%5Cmathbf%7Bp%7D%28x%2C+y%29%29%3D%5Bx-l%2C+y-t%2C+x%2Br%2C+y%2Bb%5D)解出边界框，其中![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bp%7D%28x%2C+y%29%3D%5Bl%2C+t%2C+r%2C+b%5D)（两个角点的偏移）。最终，特征图可以通过下面的式子解码，此处的![[公式]](https://www.zhihu.com/equation?tex=%5Codot)表示逐元素相乘，![[公式]](https://www.zhihu.com/equation?tex=%5Cboldsymbol%7B%5Ceta%7D_%7Bi%7D)是一个惩罚图，为每一个候选区域设置非负的惩罚得分。

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Barray%7D%7Br%7D+%5Ctilde%7BR%7D_%7Bi%7D%5E%7Bt%2B%5Cdelta%7D%3D%5Cmathcal%7BR%7D%5Cleft%28%5Cmathbf%7Bp%7D_%7Bi%7D%5Cleft%28x%5E%7B%2A%7D%2C+y%5E%7B%2A%7D%5Cright%29%5Cright%29+%3B+%5Cquad+v_%7Bi%7D%5E%7Bt%2B%5Cdelta%7D%3D%5Cmathbf%7Bv%7D_%7Bi%7D%5Cleft%28x%5E%7B%2A%7D%2C+y%5E%7B%2A%7D%5Cright%29+%5C%5C+%5Ctext+%7B+s.t.+%7D%5Cleft%28x%5E%7B%2A%7D%2C+y%5E%7B%2A%7D%5Cright%29%3D%5Cunderset%7Bx%2C+y%7D%7B%5Coperatorname%7Bargmax%7D%7D%5Cleft%28%5Cmathbf%7Bv%7D_%7Bi%7D+%5Codot+%5Cboldsymbol%7B%5Ceta%7D_%7Bi%7D%5Cright%29+%5Cend%7Barray%7D+%5C%5C)

惩罚得分的计算如下式，其中![[公式]](https://www.zhihu.com/equation?tex=%5Clambda)是一个加权系数（![[公式]](https://www.zhihu.com/equation?tex=0+%5Cleq+%5Clambda+%5Cleq+1)），![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BC%7D)则是关于目标区域![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BR%7D_%7Bi%7D%5E%7Bt%7D)几何中心的余弦窗口函数，![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BS%7D)是关于候选区域![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bp%7D%28x%2C+y%29)和![[公式]](https://www.zhihu.com/equation?tex=R_%7Bi%7D%5E%7Bt%7D)之间的相对尺度变化的高斯函数。惩罚图![[公式]](https://www.zhihu.com/equation?tex=%5Cboldsymbol%7B%5Ceta%7D_%7Bi%7D)的引入是为了阻止跟踪过程中剧烈的运动。

![[公式]](https://www.zhihu.com/equation?tex=%5Cboldsymbol%7B%5Ceta%7D_%7Bi%7D%28x%2C+y%29%3D%5Clambda+%5Cmathcal%7BC%7D%2B%281-%5Clambda%29+%5Cmathcal%7BS%7D%5Cleft%28%5Cmathcal%7BR%7D%28%5Cmathbf%7Bp%7D%28x%2C+y%29%29%2C+R_%7Bi%7D%5E%7Bt%7D%5Cright%29+%5C%5C)

考虑![[公式]](https://www.zhihu.com/equation?tex=%5Cleft%28R_%7Bi%7D%5E%7Bt%7D%2C+S_%7Bi%7D%5E%7Bt%2B%5Cdelta%7D%2C+R_%7Bi%7D%5E%7Bt%2B%5Cdelta%7D%5Cright%29)，EMM的训练损失如下图所示。其中![[公式]](https://www.zhihu.com/equation?tex=%28x%2Cy%29)表示![[公式]](https://www.zhihu.com/equation?tex=S_%7Bi%7D%5E%7Bt%2B%5Cdelta%7D)中的所有有效位置，![[公式]](https://www.zhihu.com/equation?tex=%5Cell_%7Br+e+q%7D)是用于回归的IoU损失，![[公式]](https://www.zhihu.com/equation?tex=%5Cell_%7B%5Ctext+%7Bfocal+%7D%7D)是用于分类的损失。![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bv%7D_%7Bi%7D%5E%7B%2A%7D)和![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bp%7D_%7Bi%7D%5E%7B%2A%7D)是像素级的GT图。如果![[公式]](https://www.zhihu.com/equation?tex=%28x%2Cy%29)在![[公式]](https://www.zhihu.com/equation?tex=R_%7Bi%7D%5E%7B%2A+t%2B%5Cdelta%7D)的范围内那么![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bv%7D_%7Bi%7D%5E%7B%2A%7D%28x%2C+y%29%3D1)，否则为0；![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bp%7D_%7Bi%7D%5E%7B%2A%7D%28x%2C+y%29%3D%5Cleft%5Bx-x_%7B0%7D%5E%7B%2A%7D%2C+y-y_%7B0%7D%5E%7B%2A%7D%2C+x_%7B1%7D%5E%7B%2A%7D-%5Cright.+%5Cleft.x%2C+y_%7B1%7D%5E%7B%2A%7D-y%5Cright%5D)，其中的![[公式]](https://www.zhihu.com/equation?tex=%5Cleft%28x_%7B0%7D%5E%7B%2A%7D%2C+y_%7B0%7D%5E%7B%2A%7D%5Cright%29)和![[公式]](https://www.zhihu.com/equation?tex=%5Cleft%28x_%7B1%7D%5E%7B%2A%7D%2C+y_%7B1%7D%5E%7B%2A%7D%5Cright%29)表示GT框的两个角点的坐标。此外，作者还修改了回归任务，添加了一个![[公式]](https://www.zhihu.com/equation?tex=w%28x%2C+y%29)表示中心度，定义如下：

​                                                   ![[公式]](https://www.zhihu.com/equation?tex=w%28x%2C+y%29%3D%5Csqrt%7B%5Cfrac%7B%5Cmin+%5Cleft%28x-x_%7B0%7D%2C+x_%7B1%7D-x%5Cright%29%7D%7B%5Cmax+%5Cleft%28x-x_%7B0%7D%2C+x_%7B1%7D-x%5Cright%29%7D+%5Ccdot+%5Cfrac%7B%5Cmin+%5Cleft%28y-y_%7B0%7D%2C+y_%7B1%7D-y%5Cright%29%7D%7B%5Cmax+%5Cleft%28y-y_%7B0%7D%2C+y_%7B1%7D-y%5Cright%29%7D%7D)

而损失函数公式如下：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+%5Cmathbf%7BL%7D+%26%3D%5Csum_%7Bx%2C+y%7D+%5Cell_%7B%5Ctext+%7Bfocal+%7D%7D%5Cleft%28%5Cmathbf%7Bv%7D_%7Bi%7D%28x%2C+y%29%2C+%5Cmathbf%7Bv%7D_%7Bi%7D%5E%7B%2A%7D%28x%2C+y%29%5Cright%29++%2B%5Csum_%7Bx%2C+y%7D+%5Cmathbb%7B1%7D%5Cleft%5B%5Cmathbf%7Bv%7D_%7Bi%7D%5E%7B%2A%7D%28x%2C+y%29%3D1%5Cright%5D%5Cleft%28w%28x%2C+y%29+%5Ccdot+%5Cell_%7Br+e+g%7D%5Cleft%28%5Cmathbf%7Bp%7D_%7Bi%7D%28x%2C+y%29%2C+%5Cmathbf%7Bp%7D_%7Bi%7D%5E%7B%2A%7D%28x%2C+y%29%5Cright%29%5Cright%29+%5Cend%7Baligned%7D+%5C%5C)

相比于IMM，EMM有两点改进。第一，它使用通道分离的相关性操作来允许网络显式学习相邻帧上同一个目标的相似性；第二，它采用一个细粒度的像素级监督，这有效减少了错误匹配。

#### 2.2 训练和测试

损失函数上文都已经提到了，进行训练时总的损失函数只需要将上述几个损失函数加起来即可。

在测试时，在追踪网络和检测网络的结果上各独立进行一个IoU-based NMS操作，然后再进行空间匹配操作。匹配操作如下：检测结果中和任意跟踪结果IoU大于0.5的就被匹配，然后就是轨迹操作，分为三种情况，一是如果轨迹可见度大于α则保留，二是如果有没匹配的检测对象且其可见度大于β就创建新的轨迹，三是如果连续一定数量帧都没有保留轨迹则删除该轨迹。



