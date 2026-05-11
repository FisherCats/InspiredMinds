---
title: Partially Shared CBM
summary: AAAI'26 | Concepts are shared & labled by different classes, improvement on semantic level of CBL.
date: 2026-05-11
authors:
  - admin
tags:
  - CBM
  - Concept Arrangement
  - Deep Learning
# image:
#   caption: 'Image credit: [**Unsplash**](./featured.png)'
featured: true
---
> [Paper](https://arxiv.org/abs/2511.22170)

![Arch](/PSCBM/featured.png)

## Motivation

今年来的CBM方法（LaBo, LF-CBM）通过提问大语言模型（ChatGPT）的方式来构建概念库，但还是存在一些问题：
 - 概念冗余
 - 概念与视觉样本关联性低
 - 概念不够紧凑（compactness）
![mov](/PSCBM/motivation.png)

作者认为LaBo、LF-CBM等方法为每个类别生成概念描述，使得概念库有很多冗余；而让所有类别共享全局概念又会影响模型的预测。故本文提出：
1. 多模态概念生成，构造在视觉样本中真实存在的概念
2. 通过概念融合策略减少概念冗余，在类间部分共享概念
在这里我们更加关注该方法的合并和标注策略。

## Concept Merge & Label
本文提出的概念融合方法是基于相似度计算的
![alt text](/PSCBM/alg_merge.png)

**I. 概念筛选**.  PSCBM首先计算样本与所有获取到的概念之间的相似度，得到Affinity矩阵：

$$
A_{i,j} = \cos(\Phi(x_i), \Psi(c_j))
$$

其中，每一行代表的是样本与所有概念之间的相似性分数。随后计算概念之间的相关性矩阵$Q$：

$$
Q_{i,j} = \frac{A_{:, i}^\top A_{:,j}}{||A_{:,i}|| ||A_{:,j}||}
$$

该矩阵本质上计算的是Affinity矩阵中概念与概念之间的相似性，$A_{:,i}$代表的就是第$i$个概念对所有数据样本的相似度向量。

**II. 概念融合**.  在计算得到相关性矩阵$Q$后，选择其中分数大于阈值$\tau_{merge}$的概念作为待融合的概念。根据算法流程图，下一步将选择概念组进行融合。一个概念组$S_j = \{c_2, c_4,..., c_n\}$由多个概念构成，在语义层面上说明概念组$S_j$中的这些概念与$c_j$相似度高，需要进行融合。随后从所有的概念组中选择包含概念数量最多的概念组先进行融合：

$$
c_{max} \leftarrow \argmax_{c_j \in S} |S_j|
$$

$$
\hat{S} \leftarrow \hat{S} \cup c_{max}
$$

$$
S \leftarrow \\ (c_max \cup S_{max})
$$

其中，$c_{max}$表示所含概念最多的概念组对应的概念，$S_{max}$表示$c_{max}$对应的概念组。完成概念筛选和融合后，我们就得到了最终的概念库。

**III. 概念标注**.  现在我们得到的概念库还没有标签，也就是说图像数据和概念之间的联系不确定。本文提出基于Affinity矩阵来进行概念标注：

$$
s_{i, j}=\left\{\begin{array}{ll}
1, & \text { if } y_{i} \in C_{j} \text { and } \boldsymbol{A}_{i, j}>\tau_{\text {conf }} \\
0, & \text { otherwise }
\end{array}\right.
$$

其中，$s_{i, j}$表示样本$i$对概念$j$的标签，$C_{j}$表示类别$j$相关的概念集合。