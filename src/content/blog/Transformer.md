---
title: Transformer
summary: The principle of Transformer
authors:
  - admin
tags:
  - Transformer
  - Attention mechanism
  - Deep Learning
# image:
#   caption: 'Image credit: [**Unsplash**](./featured.png)'
featured: true
---

# Transformer

## Architecture
![Structure](https://img-blog.csdnimg.cn/bc79f0495726452ea3f5704c6fbe6b60.png)

### 1. Out of Transformer: Tokenizer
训练Transformer需要根据数据集来构建一个词表，为每一个词映射到唯一的Token_ID，表示Transformer理解的词空间。此外，还需要一个Tokenizer(分词器)来将输入的sequence映射为与之对应的ID序列。例如，现有一个简化词表vocab：
```python
tokenizer.vocab = {
  "[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, 
  "你": 10, "好": 11, "，": 12, "今天": 20, "天气": 21, "如何": 22, "？": 23
} # PAD表示补全token的ID，UNK表示不在该词表里的token，BOS表示一个输入序列的开始，EOS表示一个输入序列的结束。
```
进行分词并转化为Token_ID 序列：
> "你好，今天天气如何？" $\rightarrow$ ["你", "好", "，", "今天", "天气", "如何", "？"] $\rightarrow$ [10, 11, 12, 20, 21, 22, 23]

### 2. Token Embedding
在这一部分将输入的Token转化为对应的词向量，作为每个Token的高维特征。

Transformer首先初始化一个`nn.Embedding`对象并设置其大小以及embedding维度：
```python
import torch.nn as nn

vocab_size = 11  # 词表大小
d_model = 512       # 嵌入维度
padding_idx = 0     # 填充符号的索引

# 定义嵌入层（自动处理填充符号的梯度）
token_embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=d_model,
    padding_idx=padding_idx
)
```
Token Embedding这一层是需要训练的。在training时，Transformer会将输入的sequence中每一个token按照其相应的token id提取出对应的在 `token_embedding` 中的词向量(词嵌入)，拼接为当前sequence的输入embedding，来进行后续的计算。

继续上一节的例子，这里输入的Token_ID序列为 `[2, 10, 11, 12, 20, 21, 22, 23, 3]`。与原始序列相比，增加了BOS和EOS，token_embedding会执行
```python
input = token_embedding[Token_ID] 
```
来选择对应ID的token embedding继续训练。
### 3. Positional Embedding
这一部分对输入的sequence加上一个位置编码，使模型能够理解到token的顺序先后。每一个token都会加一个对应的位置编码，维度与token embedding维度相同。在原始的transformer中，positional embedding是不被训练的，固定的。transformer对输入的sequence的奇数位置和偶数位置分别进行sin编码和cos编码。

### 4. Multi-head Self-Attention
Transformer中的**多头自注意力(Multi-Head Self-Attention)**是模型捕捉序列内部复杂依赖关系的核心机制。它通过并行多个“注意力头”，使模型能够同时关注不同位置和语义层面的信息。

**关键优势**：
1. 并行性：多个头独立计算注意力，提升计算效率。
2. 多样性：不同头可关注不同模式（如局部依赖、长距离关联、语法角色等）。

在完成Token Embedding 和 Positional Embedding操作之后，我们就得到的sequence序列的一个高维特征，形状为 [1, 9, d_model]. 下面我们将进行自注意力来计算词与词之间的关系。首先梳理一下MSA的结构：

- W_Q, W_K, W_V $\rightarrow$ **QKV 映射矩阵**
- proj $\rightarrow$ 最终线性变换

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 定义线性投影矩阵
        self.W_Q = nn.Linear(d_model, d_model)  # 实际拆分为h个头
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
```

MSA 模块就是利用这些矩阵之间的计算得到token与token之间的注意力关系。MSA的计算公式为：
$$
\text{Attention}(Q, K, V) = \text{Softmax}(\frac{QK^\top}{\sqrt{\text{d\_k}}})\cdot V
$$
其中，Q、K、V分别代表输入序列$x$分别经过W_Q、W_K、W_V映射后的特征。MSA模块的 **forward** 过程：

```python
def forward(self, X, mask=None):
    batch_size, seq_len, _ = X.shape
    
    # 生成Q/K/V并分头
    Q = self.W_Q(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    K = self.W_K(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    V = self.W_V(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    # 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
    # 应用Mask
    if mask is not None:
        # mask形状需为(batch_size, 1, seq_len, seq_len)或可广播的形状
        scores = scores.masked_fill(mask == 0, -1e9)  # 将mask中0的位置设为 -inf
    
    attn_weights = F.softmax(scores, dim=-1)

    # 加权求和并合并多头
    output = torch.matmul(attn_weights, V)
    output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    
    # 最终线性变换
    return self.proj(output)
```
上面这个过程可以通过这个图更加直观理解：
![Alt text](/Transformer/MSA.png)
在上图中，绿色的向量代表输入的单个token，橙色、粉色、红色向量分别表示当前token经过W_K, W_V, W_Q之后得到的向量，即上述公式中的Q，K，V。之后MSA会求Q与K的点乘，得到每个token与其他token的注意力矩阵(图中蓝色部分)，最终再与V进行计算。

#### **Attention Mask**
屏蔽无效位置：防止模型关注填充符（Padding Tokens）或未来的信息（解码时避免看到后续词）。例如输入序列 `[I, love, NLP, [PAD], [PAD]]`，mask标记`[1,1,1,0,0]`，在MSA计算时会将PAD token得到的值抹去($-\infin$)

### 5. Encoder
编码器部分由N个相同结构的Block组成([text])[Structure]，每个Block包含：
- 一个MSA模块
- 一个FFN模块
- LayerNorm层
```python
# PyTorch示例
attn_output = self.attention(x)
x = self.norm1(x + attn_output)  # 残差连接后接层归一化
ffn_output = self.ffn(x)
x = self.norm2(x + ffn_output)
```
Encoder接收一个输入句子列表, 即sequence. 在forward的过程中会生成注意力mask

### 6. Decoder

<!-- #### Masked Multi-Head Self-Attention -->
解码器部分由两个多头注意力模块和一个FFN模块构成。
```python
class Decoder(nn.Module):
    def __init__(self, d_model, num_head, embd_size):
        self.MSHA1 = MultiHeadAttention(d_model, num_head)
        self.linear1 = nn.Linear(d_model*num_head, embd_size)
        self.norm1 = nn.LayerNorm(embd_size)

        self.MSHA2 = MultiHeadAttention(d_model, num_head)
        self.linear2 = nn.Linear(d_model*num_head, embd_size)
        self.norm2 = nn.LayerNorm(embd_size)

        self.ffn = nn.Sequential(
            nn.Linear(embd_size, d_model),
            nn.ReLU(),
            nn.Linear(d_model, embd_size)
          )
        self.norm3 = nn.LayerNorm(embd_size)
    
    def forward(self, x, encoder_out, attn_mask_1, attn_mask_2):
        out_1 = self.MSHA1(x, x, attn_mask1)
        out_1 = self.linear1(out_1)
        out_1 = self.norm1(x + out_1)

        out_2 = self.MSHA2(out_1, encoder_out, attn_mask_2)
        out_2 = self.linear2(out_2)
        out_2 = self.norm2(out_2 + out_1)

        return self.norm3(self.ffn(out_2) + out_2)

```

## The Training phase of Transformer

## The Inference phase of Transformer
<!-- 模型的推理过程 -->




# Vision Transformer

## ViT Structure
```python
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool="token", 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, init_values=None, class_token=True, no_embed_class=False, fc_norm=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, weight_init="", embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, prompt_length=None, embedding_key="cls", prompt_init="uniform", prompt_pool=False, prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init="uniform", head_type="token", use_prompt_mask=False):
                 ······
```
### 1. Patch Embed
```python
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        ······
```
patch_embed layer主要有两部分构成：
- ```self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)```
- ```self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()```

在这一层，将$224\times 224$的图像首先通过2维卷积得到高维图像embedding[bz, $N_p$, embed_dim]，记作$x$.  $N_p = (\frac{img\_size}{patch\_size})^2$, 代表patch的数量，也称做一个token；embed_dim表示高维空间维度。最后进行layer_norm操作，图像embedding形状不变，数值进行归一化。

### 2. Positional Embed
```self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)```
位置嵌入初始为一个随机tensor，其形状为 [1, $N_p$+1, embed_dim]. 在ViT中，这个位置编码是可学习的，模型通过训练学到不同patch的位置关系。

在Forward过程中，$x$会与位置编码进行相加，得到带有位置信息的图像embedding，形状不变。

### 3. Residual Blocks
```python 
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, init_values=None, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
```
在这里会执行ViT的大部分计算，ViT模型根据其规模所包含的block数量不同，但每一个block的结构相同。例如，ViT-B包含12个block，ViT-L包含24个block。

每个block包含一个多头自注意力（MSA）和一个多层感知机（MLP）:
- 多头自注意力MSA
  ```python
  class Attention(nn.Module):
      def __init__(self, dim, num_heads=12, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
  ```
  MSA主要包含: qkv_proj(linear), attn_drop(Dropout), proj(linear)

  在MSA的forward过程中，$x$会首先通过qkv_proj层得到qkv特征，分别作为query，key和value，按照如下公式进行计算：
  $$
  \text{Attention(Q,K,V)} = \text{softmax}(\frac{QK^\top}{\sqrt{d_k}})V
  $$
  $QK^\top$计算得到所有patch的注意力矩阵（激活图），每个头都计算得到 $\text{num heads}$个注意力矩阵。经过$\text{softmax}$后再与V计算得到注意力特征，赋值给$x$，形状为 [bz, $N_p+1$, dim(768)]. 最后$x$经过```proj```返回映射后的特征。

- 多层感知机MLP
  ```python
  class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        ...
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
        ...
  ```
  MLP的结构非常简单，主要由两个线性层和一个激活层构成。这两个线性层可以看作是up_proj和down_proj映射。输入的$x$依次经过```fc1, act1, drop1, fc2, drop2```，得到映射后的特征，$x$维度不变。

### 4. Norm Layer

```python
class LayerNorm(Module):
      def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True,device=None, dtype=None) -> None:
```

层归一化，沿特征维度对输入$x$进行归一化：
$$
\mu = \frac{1}{D}\sum_{i=1}^D x_{b,n,i} \\  

\sigma = \sqrt{\frac{1}{D}\sum_{i=1}^D (x_{b,n,i} - \mu)^2} \\

y_{b,n,i} = (\frac{x_{b,n,i} - \mu}{\sigma + \epsilon})\cdot \gamma_i + \beta_i
$$
其中，$\gamma$和$\beta$是可学习参数，$\epsilon$是为数值稳定性添加的小常数（如1e-5）。
