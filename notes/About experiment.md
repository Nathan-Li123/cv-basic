# 实验

### 一、评估方法

##### mAP

比如Pascal VOC 一般使用 mAP[0.5] 来进行评估，而MSCOCO有mAP[0.5]还有mAP[0.5，0.95]。还有Caltch101和256数据集，具体见[VGG](./VGG.md)笔记中notes的相应部分。

##### top-n错误率

ILSVRC会使用top-1和top-5

### 二、实验方法

##### 数据实验

总归需要有一个论文模型跑在数据集上得到结果的实验，可以设定不同的超参进行对比。同时可以和同类型的模型，或者本论文的改进所基于的原模型相应数据进行比较。主要数据集有ILSVRC、PASCAl VOC、MS COCO等，数据集测试可以放在论文中也可以将结果放在附录中。

##### 消融实验

消融实验就是逐个验证论文中的重要贡献点或贡献点组合。消融实验和单纯的数据测试可以同时进行，不断地加入论文所提出的改进来见证精度或者速度的提升。

##### 与其他模型进行结合

比如VGG的论文中有一组实验就是将VGG和一些其他的模型结合起来进行测试。

##### 数据集规模实验

还可以进行不同数据大小的训练实验。





