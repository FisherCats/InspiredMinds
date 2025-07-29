---
title: Convolutional Neural Network 
summary: The principle of CNN
date: 2025-07-29
authors:
  - admin
tags:
  - CNN
  - Deep Learning
# image:
#   caption: 'Image credit: [**Unsplash**](./featured.png)'
featured: true
---

## 1. Intro
卷积神经网络（Convolutional Neural Network，CNN）是一种在计算机视觉领域取得了巨大成功的深度学习模型。它们的设计灵感来自于生物学中的视觉系统，旨在模拟人类视觉处理的方式。在过去的几年中，CNN已经在图像识别、目标检测、图像生成和许多其他领域取得了显著的进展，成为了计算机视觉和深度学习研究的重要组成部分。

我们希望模型能够不受图像中的物体在图中的位置、大小等几何变换的影响，正确的做出判断，这一特点就是不变性。CNN在卷积层中的卷积操作可以捕捉物体在图像中的局部特征而不受位置等变换的影响。

## 2. 为什么使用卷积神经网络

使用全连接网络处理图像时，有非常明显的缺点：
- 将图像展开为向量会丢失空间信息
- 参数过多处理效率低下，训练困难
- 大量的参数也会容易使得网络快速过拟合

而CNN通过卷积层可以很好的解决这些问题

## 3. 卷积操作
### 3.1 卷积操作是什么？
卷积（Convolution）从数学层面上来看十分抽象，但通过图像可以很直观的理解：
![simple_conv](/CNN/simple_conv.webp)   
从上图可以看出，卷积操作通过一个滑动窗口与输入图像（下方蓝色矩阵）进行某种操作，提取出图像的卷积后的特征（上方绿色矩阵）。 卷积操作其实就是每次取一个特定大小的矩阵$K$（蓝色矩阵中的阴影部分），然后将其对输入图像$X$（图中蓝色矩阵）依次扫描并进行内积的运算过程，最后得到卷积后的特征$F$。  

我们将K称之为卷积核（Convolutional Kernal / filter），可以是一个也可以是多个，卷积后得到的结果$F$称为特征图（Feature Map）。

### 3.2 卷积操作的计算
下面通过直观的图像来解释卷积的计算过程，首先是单通道单卷积核，了解之后可以扩展到多通道多卷积核的情况。

#### 3.2.1 单通道单卷积核
如下图所示，现有一```[5, 5, 1]```形状的灰度图，需要用右边的卷积核对其进行卷积处理，过程是怎么样的呢？   
![single_channel_single_kernal](/CNN/single_channel_single_kernal.png)  
根据上一节介绍，卷积操作要用卷积核对输入图像进行滑动计算的过程，那么可以得到：
![single_channel_single_kernal_cal](/CNN/single_channel_single_kernal_cal.png)  
所以卷积图中的每一个元素，都是通过卷积核与输入图像对应位置的内积+bias得到的，通过卷积也将输入图像的维度进行了缩减，得到形状为```[3, 3, 1]```的特征图。

#### 3.2.2 单通道多卷积核
单通道多卷积核与单通道单卷积核相比卷积核更多，所以最后得到的特征图也更多：
![single_channel_multi_kernal](/CNN/single_channel_multi_kernal.png)
![single_channel_multi_kernal_cal](/CNN/single_channel_multi_kernal_cal.png)   

#### 3.2.3 单通道多卷积核
在实际生活中，我们遇到彩色的图像的情况更多，这样就需要卷积层能够处理多通道的数据。在这里需要注意的是，卷积层中对于一个多通道的输入图像，每一个卷积核也分多个通道去进行卷积操作。   
![multi_channel_single_kernal](/CNN/multi_channel_single_kernal.png)
![multi_channel_single_kernal_cal](/CNN/multi_channel_single_kernal_cal.png)   

#### 3.2.4 多通道多卷积核
通过上面的介绍，很容易扩展到多通道多卷积核的情况：  
![multi_channel_multi_kernal](/CNN/multi_channel_multi_kernal.png)
![multi_channel_multi_kernal_cal](/CNN/multi_channel_multi_kernal_cal.png)  
通过上面的计算，我们可以发现：  

- 输入图像有多少个通道，其对应的卷积核就有多少个通道
- 用k个卷积核对输入图像进行处理，那么最后得到的特征图就一定会包含有k个通道

## 4. 卷积结构类型

### 4.1 标准卷积
标准卷积就是上述讨论过的卷积计算流程，每一个卷积核与输入的图像进行点积存操作得到特征图：   
![standard_conv](/CNN/standard_conv.jpg)

### 4.2 分组卷积
分组卷积是减少卷积计算量的一种方法。因为最后的特征图的数量越来越多，卷积核的通道数也越来越多，导致计算量逐步增加。分组卷积是将输入特征分组，每个卷积核只需要与其中的一组进行计算，每个卷积核的通道数只与每组特征图的通道数相同，下图是分组为2的例子：  
![group_conv](/CNN/group_conv.png)
![group_conv_result](/CNN/group_conv_result.png)

### 4.3 深度卷积
深度卷积是分组卷积的延伸，深度卷积将每个输入特征图分为一组，这样对应的卷积核的通道数为1。   
![deep_conv](/CNN/deep_conv.png)  
这样直接对于每个特征图进行标准的卷积操作。但存在一个问题，就是每个特征图由对应的卷积核计算，那么其中的信息只在对应卷积核得到的特征图中保存，也就是说第一个输出特征仅依赖于第一个输入特征，这样的模式会在网络的更深层次中持续。而只有一组的卷积操作中，卷积核是与所有的输入特征图进行计算，运用了全部的输入特征信息进行计算，有更好的全局表达能力。为了解决这个问题，人们提出在深度卷积后面加上一个点对点卷积（pointwise convolution）来融合特征的全局信息，这样的卷积模式也称为深度可分离卷积（depthwise-separable convolution）。   
![depthwise_separable_conv](/CNN/depthwise_separable_conv.png)

### 4.4 空洞卷积

### 4.5 转置卷积

## 5. 感受野（Reciptive Field）
感受野指的是卷积神经网络每一层输出的特征图上每个像素点映射回输入图像上的区域大小。神经元感受野的范围越大表示其能接触到的原始图像范围就越大，也意味着它能学习更为全局，语义层次更高的特征信息；相反，范围越小则表示其所包含的特征越趋向局部和细节。从结构上来看，感受野的大小与第一层卷积的卷积核大小很相关。   
深度卷积神经网络中靠前的层感受野较小，提取到的是图像的纹理、边缘等局部的、通用的特征；靠后的层由于感受野较大，提取到的是图像更深层次、更具象的特征。因此在迁移学习中常常会将靠前的层的参数冻结（不参与训练，因为他们在迁移到新的场景之前已经具备了提取通用特征的能力），来节省训练的时间和算力消耗。

