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

# Convolutional Neural Network

## Intro
卷积神经网络（Convolutional Neural Network，CNN）是一种在计算机视觉领域取得了巨大成功的深度学习模型。它们的设计灵感来自于生物学中的视觉系统，旨在模拟人类视觉处理的方式。在过去的几年中，CNN已经在图像识别、目标检测、图像生成和许多其他领域取得了显著的进展，成为了计算机视觉和深度学习研究的重要组成部分。

我们希望模型能够不受图像中的物体在图中的位置、大小等几何变换的影响，正确的做出判断，这一特点就是不变性。CNN在卷积层中的卷积操作可以捕捉物体在图像中的局部特征而不受位置等变换的影响。

## 为什么使用卷积神经网络

使用全连接网络处理图像时，有非常明显的缺点：
- 将图像展开为向量会丢失空间信息
- 参数过多处理效率低下，训练困难
- 大量的参数也会容易使得网络快速过拟合

而CNN通过卷积层可以很好的解决这些问题

## 卷积操作是什么
卷积（Convolution）从数学层面上来看十分抽象，但通过图像可以很直观的理解：
![conv](/CNN/conv.gif)

图中左侧的蓝色矩阵代表输出图像的三个通道，每个通道都是一个 $H \times W$ 的矩阵，左侧第二列矩阵表示在这三个通道分别施加的
