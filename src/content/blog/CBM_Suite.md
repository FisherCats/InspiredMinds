---
title: Rethinking Concept Bottleneck Model:From Pitfalls to Solutions
summary: CVPR'26 | Rethinking the effectiveness of CBMs, is it makes interpretable decisions? and How?
date: 2026-05-12
authors:
  - admin
tags:
  - CBM
  - Deep Learning
# image:
#   caption: 'Image credit: [**Unsplash**](./featured.png)'
featured: true
---
> [Paper](https://arxiv.org/abs/2603.05629)

## Motivation

<img src="/CBM_Suite/mov.png" alt="mov" width="60%" />

虽然现有的 CBM 方法从人类可理解的层面解释模型的决策，但仍存在一些局限。现有方法往往关注模型的可解释性和性能方面，没有关注到更深层次的重要问题。本文首先对现有 CBM 方法在解释性和模型结构上的问题进行了阐述，随后引出本文的方法 CBM-Suite

## Concept Set Relevance

语言引导的概念瓶颈模型（LG-CBM，如LF-CBM、LaBo、LM4CV、VLGCBM等）通过提问大语言模型来获取与目标类别相关的概念描述，但这些概念中存在对模型分类没有帮助的概念，例如背景、环境、习性等。现有方法通过设计不同的筛选方式来进一步选择视觉和语义层面一致的概念描述，作为模型的概念池（concept pool）。本文作者提出通过计算概念与样本之间的熵，作为衡量概念集与数据集中样本是否匹配的依据。具体来说，给定一个样本$x_i$, $c_i \in \mathbb{R}^K$是该样本对应的概念激活分数，计算公式为：
$$
c_i = \frac{f_I(x_i)\cdot f_T(\mathbb{C}) - \mu}{\sigma}
$$

其中，$f_I$和$f_T$分别表示图像编码器和文本编码器，$\mathbb{C}$ 表示概念池中的概念集合。$\mu$和$\sigma$表示概念激活分数的均值和标准差。由此可以得出概念激活分数在下游任务数据集上的分布，作者对比了无关概念集与相关概念集分布的差异：


<img src="/CBM_Suite/distribution_diff.png" alt="distribution_diff" width="70%" />

从上图中可以看到，无关概念集在目标数据集上的概念分数分布接近于正态分布，而相关概念集的概念分数分布更加集中，呈现“更高、更窄”的特性。作者认为，好的概念解释对应的概念分数应当是稀疏的，激活概念与未激活概念的差值大。基于此，作者提出利用概念激活分数的熵来评估概念集与当前任务的相关性：
$$
c_{i,k}' = \frac{\exp (c_{i,k})}{\sum_{j=1}^K \exp (c_i,j)}
$$

$$
H(c_{i,k}') = -\sum_{j=1}^K c_{i,j}' \log c_{i,j}'
$$

首先对所有概念激活分数求 softmax，然后计算它们的熵$H$。

 - 低熵：少数概念占据主要概率，说明概念激活集中，解释更稀疏、更可能有意义。
 - 高熵：很多概念激活差不多，说明概念集对图像没有清晰响应，更像噪声。

完成熵值计算后，本文从两个层面评估概念集的相关性。

1. Task-agnostic：评估概念集合对整体数据集的相关性，计算所有样本概念熵值的平均。
2. Task-specific：评估概念集合对每个类别的相关性，计算所有类别概念熵值的平均。
   

在主流的 CBM 结构中，目标样本由图像编码器（CNN、VIT等）提取其图像特征，然后通过概念瓶颈层（CBL）将图像特征映射到概念分数空间，得到图像对所有概念的激活分数，最后由分类器层完成分类。这种情况下，CBM 中所包含的概念是人为确定的，CBL 中的神经元被人为指定代表某个概念。从网络结构上来看，其实就是在Backbone 和 Classifier 之间加入一层 Linear 或 MLP，构建可解释的中间表示。因此，如果没有较强的可解释性约束，CBL 层输出的中间表示可能不足以提供解释性，而概念集是否与目标数据集相关也不会影响到模型的性能。

## Linearity Problem
现有的 LG-CBM 方法往往直接利用 Linear 或 MLP 构建 CBL，将图像特征映射到概念分数空间。作者认为这样的设计绕过了概念瓶颈的结构，即使有多个线性映射，它们最终可以融合成一个单一的线性变换，模型整体退化为线性变换的分类模型，导致模型在实际分类的过程中没有真正利用到概念瓶颈。

而早期的 CBM 方法中有提到使用非线性激活层来构建 CBL。例如原始[CBM](https://arxiv.org/abs/2007.04612)方法中提出在构建概念分数到类别空间的映射时，插入非线性激活层 ReLU 来构建 CBL：

```text
# 原始CBM方法在OAI数据集上的工作流程
image -> ResNet-18 -> Linear -> concepts
concepts -> MLP(50 hidden units, 50 hidden units) -> KLG grade
```

本文作者重申该观点，认为 CBM 应当避免完全构建为线性变换的结构，并通过实验验证了纯线性变换结构会损害模型的可解释性。

但仔细来看，二者又有不同之处。CBM 提出利用 ReLU 来构造 CBL，是出于概念组合和层级的假设，让 CBL 层能够表达复杂的概念关系。例如在 OAI 数据集上的分级任务，标签 $y$ 往往不是概念的简单线性加权，而可能依赖：

- 概念组合：两个症状一起出现才强烈指向某个等级；
- 阈值效应：某个概念超过一定严重程度后，标签变化很大；
- 饱和效应：概念继续变严重，但标签等级不再线性增加；
- 非对称关系：轻微骨刺和严重关节间隙变窄的影响不同。

在原始CBM中，线性分类器只能学到概念之间线性的关系对分类的影响，而利用非线性激活层可以学习如上复杂的概念关系以及如何据此进行分类。此外，原始CBM 方法中在 OAI 数据集上的实验表明了带有非线性层的CBL 设计可能具有更好的干预性能。而在另一个数据集 CUB 上，原始 CBM 方法还是应用了 Linear 作为 CBL 层，这也说明构建 CBM 时非线性层并不是必要的，这一点在本文当中没有说明。

本文作者认为纯线性变换结构会影响模型的可解释性，导致模型对概念集不敏感，即使给定随机的概念集，模型也能够做出正确的预测：

![linearity problem](/CBM_Suite/linearity_problem.png)

这种情况在 LG-CBM 方法中比较明显。由于缺乏显式概念标签，模型直接将提取到的概念的文本特征堆叠作为概念瓶颈（LaBo，LM4CV），或通过 CLIP 软标签的形式来约束 CBL 输出与 CLIP 一致，保证模型能够提供合理的解释。但在随机概念的情况下，CLIP 得到的概念激活矩阵也是无意义的，CBM 依据随机的概念进行分类。在交叉熵损失的引导下，模型并不知晓概念是否与数据集中的样本相关联，所以需要额外的方式去判断。

此外，本文作者进行实验发现，具有非线性层的CBM对概念集更加敏感：

![acc_comp_linear_non_linear](/CBM_Suite/acc_comp.png)

我们从图中可以看到，Linear CBM 的性能受概念集变化的波动比较小，Non-Linear CBM 受概念集变化的波动较大。作者认为Non-Linear CBM 会依赖于有意义的概念表示来进行预测，Linear CBM 可以通过简单的线性变换完成分类。

但这里的核心区别在于是否使用非线性层，而激活非线性层的神经元需要输入达到一定阈值，这是它的特性（如ReLU）。本文方法通过计算CBL输出与监督 VLM 模型 SAIL 输出的激活分数之间的MSE损失引导模型的概念学习过程，模型是否学习到正确的类别-概念关系关键在于 SAIL 的预训练知识中是否含有概念类别之间的正确联系。因此可以进一步猜想性能下降的本质原因：

对于 relevant concepts，文本概念与图像类别之间更可能存在语义对应关系，因此图像-概念相似度或概念预测分数更容易形成有效的、可区分的激活模式。非线性层，如 ReLU，在这种情况下更多保留任务相关的正向信号，使后续分类器仍能从概念表示中获得有用信息。

而对于 irrelevant concepts，尤其是 random words，这些概念很难在语义上描述目标样本。图像与随机词之间的相似度往往缺少稳定正向语义关联，概念分数可能更低、更分散，或者更接近噪声。经过 ReLU 等非线性激活后，较多神经元会被截断为零，导致 concept bottleneck 中可传递的信息减少。于是，模型后续分类器接收到的表示不仅语义无关，而且信息量也被进一步压缩，分类性能下降更明显。从上图中我们也可以看到，从 Linear CBM 到 Non-Linear CBM，模型在 relevant / irrelevant 概念集上性能的变化，可以明显观察到在 relevant 概念集上，引入非线性层带来的性能下降要比在 irrelevant 概念集的幅度低很多，一定程度上可以佐证我们的猜测。

## Method

![arch](/CBM_Suite/arch.png)

### Concept Alignment

本文所提出方法 CBM-Suite的模型结构如上图所示，在 CBL 中插入了非线性层。本文采用与 LF-CBM 相同的范式，通过 VLM 的激活分数矩阵引导模型的概念学习过程。具体来说，本文计算CBL层输出的概念分数与VLM的激活分数之间的 MSE 损失：

$$
L_{CE} = \frac{1}{N}\sum_{i=1}^N ||\hat{c}_i - c_i||^2
$$

### Teacher-Guided Training

为了弥补非线性层加入带来的模型性能损失，本文额外设置一个教师，直接基于图像特征完成分类，通过蒸馏教师和 CBM 输出的 logits 来提升模型的分类性能。利用交叉熵损失指导训练过程：

$$
L_T = -\sum_{i=1}^N y_i\log(\text{softmax}(W_t\cdot f(x_i)))
$$

其中 $W_t$ 表示教师分类器，$f(x_i)$ 表示图像特征。训练好教师后，结合蒸馏损失、交叉熵损失和稀疏损失指导分类器的训练过程：

$$
L_C = \alpha \cdot [-\sum_{i=1}^N y_i\log(\text{softmax}(W_f \cdot \hat{c_i})) + \lambda ||W_f||_1 +(1-\lambda) ||W_f||^2 ] \\ +  \beta \cdot T^2 \sum_{i=1}^N z_i(\log (\frac{W_f \hat{c_i}}{T} - \log \frac{z_i}{T}))
$$

## Experiments

CBM-Suite 与 其他 SOTA 方法的性能对比：

![maintab](/CBM_Suite/maintab.png)

应用不同的 backbone 和 VLM 对模型性能的影响（IN100）：

![backbone_vlm_pair](/CBM_Suite/backbone_vlm_pair.png)
