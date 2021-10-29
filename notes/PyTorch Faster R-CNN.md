# PyTorch Faster R-CNN

### 一、PyTorch自带Faster R-CNN

#### 1.1 预训练完成的模型

PyTorch 本身就有Faster R-CNN的实现，因此首先最简单地实现了一个backbone为ResNet的Faster R-CNN的测试，测试传入一张图片进行识别，之后在图片的相应位置标上bbox以及预测的类别。具体代码见[Faster R-CNN.py](../codes/faster_rcnn_01.py)，注释还是比较详细的。加载这个模型很简单，只不过需要一些加载时间，通过以下代码就可以实现：

```python
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model = model.to(device)
model.eval()
```

可以看到下载的模型是一个faster rcnn模型，他的backbone使用的是RestNet50，并且模型包含FPN，可选参数pretrained说明下载下来的模型是训练好的。eval函数是在模型测试之前的时候使用的一个函数。

整体效果流程如下，首先传入一张照片：

<img src="E:/李云昊/国科/computer-vision/notes/img/input.jpg" style="zoom:70%;" />

由于PyTorch自带的Faster R-CNN是用MS COCO数据集训练的，因此他分类也是按照MS COCO的设定进行分类。经过代码运行最终得到的结果如下：

<img src="E:/李云昊/国科/computer-vision/notes/img/output.png" style="zoom: 60%;" />

可以看到图片中标上了很多bbox，这个结果是为所有检测分数大于0.5的对象标上了bbox，可以说定位还是相当准确的，并且识别出的对象也非常多。

#### 1.2 训练模型

##### 准备工作

除了需要安装pytorch和torchvision外，还需要安装pycocotools（这个东西在windows上安装比较麻烦，建议在Linux上安装）。然后下载PASCAL数据集（也可以使用COCO或者自己创建的数据集，但是格式有标准），数据集目录如下：

<img src=".\img\PASCAL数据集目录.jpg" style="zoom:80%;" />

##### 定义模型

使用pytorch自带的faster rcnn模型是最容易的，代码如下：

```python
model = detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=20, pretrained_backbone=True)
```

##### 数据增强

在图像输入到网络前，需要对其进行数据增强。这里需要注意的是，由于Faster R-CNN模型本身可以处理归一化（默认使用ImageNet的均值和标准差来归一化）及尺度变化的问题，因而无需在这里进行mean/std normalization或图像缩放的操作。
由于from torchvision import的transforms只能对图片进行数据增强，而无法同时改变图片对应的label标签，因此我们选择使用torchvision Github项目中的一些封装好的[用于模型训练和测试的函数](https://github.com/pytorch/vision/tree/master/references/detection)。

##### 训练模型



### References

* [使用pytorch训练自己的Faster-RCNN目标检测模型](https://www.cnblogs.com/wildgoose/p/12905004.html)

