---
title: KVCache
summary: How KV Cache works for Transformers
date: 2025-02-28
authors:
  - admin
tags:
  - Transformer
  - Attention mechanism
# image:
#   caption: 'Image credit: [**Unsplash**](./featured.png)'
---

# KVCache
KVCache（Key-Value缓存）是Transformer模型在自回归生成任务​（如文本生成、机器翻译）中用于加速**推理**的一种优化技术。它通过缓存历史时间步的Key和Value向量，避免重复计算，将自注意力（Self-Attention）的计算复杂度从O(n²)降低到O(n)，显著提升长序列生成的效率。(from DP-R1)

其核心思想还是**以空间换时间**，通过缓存K、V，可以避免Linear层的重复计算，从而降低计算量。在transformer的推理过程中， encoder部分接收用户输入的prompt进行编码，得到其高维特征；decoder部分会持续输出模型预测的token，每个token表示一个中文字或英文单词。此外，当模型输出完一个token后，会将其与之前输出的token拼接在一起，然后再次送入到decoder中计算。

KVCache 技术在Transformer架构中可以优化的点分为两个层面。一个就是对于KV变量的缓存，分别是在编码器和解码器求注意力计算的过程中；另一个是可以应用在注意力激活矩阵中，对于之前的token的矩阵元素的缓存。

**注意**，KVCache技术仅在decoder中应用。
<!-- ![Alt text](transformer_structure.png) -->
## 1. Cross-attention KVCache
首先是交叉注意力部分的KVCache。这部分的注意力要求encoder的输出与decoder中经过MSA的输出进行attention计算。在上图中可以看到，encoder的输出作为decoder中第二个MSA模块的K，V，与decoder第一个MSA模块得到的特征进行cross-attention计算。

在推理阶段，decoder会不断的输出token，不断重复其中的attention计算。而每次计算中K、V是不变的，都来自于encoder的输出。故将此K、V缓存可以提高其推理速度。

## 2. Self-attention KVCache
另一个可以进行KVCache优化的点在decoder中的第一个MSA模块，这一部分直观来看可以称作QKVCache，因为可以将QKV三个变量都进行缓存来降低计算量。在推理阶段，之前预测的token会与当前预测得到的token进行拼接，再次作为decoder的输入，经过decoder计算得到下一个预测token。在这个循环的过程中，decoder的输入首先都会经过MSA计算自注意力。而之前预测得到的token的QKV都是已经计算过的，可以将其缓存，然后在需要计算的时候直接调用。这样模型在每一次decoder的forward过程中，仅需要计算上一次生成的token的QKV即可，然后将其保存便于下一次计算直接调用，提高模型的推理速度。

## 3. Final
这样的KVCache过程仅仅是缓存了输入变量与W_Q, W_K, W_V线性层计算后的embeddings，对于后续QKV之间的计算并没有提出优化，不过顺着transformer的结构和推理流程，attention激活矩阵的部分应该可以再进行缓存优化。
