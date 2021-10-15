# Pytorch

*Pytorch官方教程：../resources/cv basic/Pytorch官方教程中文版.pdf*

### 一、Functions

##### Conv2d

torch.nn.Conv2d(**in_channels, out_channels, kernel_size**, *stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None*) 

产生一个二维卷积核，同理Conv1d、Conv3d也就是产生一维卷积核、三维卷积核。主要参数为前三个，其中kernel_size可以是一个int也可以是一个tuple，如果是一个tuple则代表长和宽，如果是一个int则代表长宽均是该整型。

##### Linear

torch.nn.Linear(**in_features, out_features**, bias=True, device=None, dtype=None)

进行一个线性变换。就是神经网络中从输出层到隐藏层极其类似操作的实现。

##### MaxPool2d

torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

建立一个二维最大值池，同样的MaxPool1d、MaxPool3d就是一维和三维池，kernel_size和Conv2d的参数一致，可以是tuple或int。



