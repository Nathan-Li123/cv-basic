# Pytorch

*Pytorch官方教程：../resources/cv basic/Pytorch官方教程中文版.pdf*

### 一、Functions

##### Conv2d

```python
torch.nn.Conv2d(**in_channels, out_channels, kernel_size**, *stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None*) 
```

产生一个二维卷积核，同理Conv1d、Conv3d也就是产生一维卷积核、三维卷积核。主要参数为前三个，其中kernel_size可以是一个int也可以是一个tuple，如果是一个tuple则代表长和宽，如果是一个int则代表长宽均是该整型。

##### Linear

```python
torch.nn.Linear(**in_features, out_features**, bias=True, device=None, dtype=None)
```

进行一个线性变换。就是神经网络中从输出层到隐藏层极其类似操作的实现。

##### MaxPool2d

```python
torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
```

建立一个二维最大值池，同样的MaxPool1d、MaxPool3d就是一维和三维池，kernel_size和Conv2d的参数一致，可以是tuple或int。

### 二、重点概念

*以下问题来自[Pytorch极简入门路线](../resources/cv basic/PyTorch极简入门路线.pdf)*

#### 2.1 模型是什么：torch.nn.Module

官方定义说 torch.nn.Module 是所有神经网络单元的基类，module里面还可以包含submodule ( 比如Conv2d其实就是创建了一个submodule )。一般来说自主定义的一个继承nn.module的类就是所定义的神经网络。

#### 2.2 如何加载数据：torch.utils.data.DataLoader

```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, 	                    collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, *,                      prefetch_factor=2, persistent_workers=False)
```

##### 重要参数

* dataset：这个就是PyTorch已有的数据读取接口（比如torchvision.datasets.ImageFolder）或者自定义的数据接口的输出，该输出要么是torch.utils.data.Dataset类的对象，要么是继承自torch.utils.data.Dataset类的自定义类的对象。是唯一一个必选参数。
* batch_size：设置每一批加载多少样本数据。
* shuffle：在每批中将数据打乱，通常在训练数据集中会使用。
* num_workers：设置用多少个子进程来读取数据，如果是0的话就是只有主进程读取，参数必须大于等于0。
* timeout：设置数据读取时间的超时限制。

##### 示例

以加载CIFAR10数据集为例：

```python
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)
```

#### 2.3 如何训练: backward, optimizer, schduler

* backward：反向传播
* optimizer：反向传播时设置的参数更新器
* schduler：学习率调整策略

#### 2.4 重要函数用途

##### optimizer.zero_grad()

将梯度初始化为0，反向传播的时候再次进行梯度下降。

##### loss.backward()

进行反向传播。

##### optimizer.step()

这个方法就是令optimizer去更新所有的参数。



