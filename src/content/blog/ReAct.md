---
title: ReAct：Synergizing Reasoning and Acting in Language Models
summary: ICLR'23 | Make agents think as human
date: 2026-07-28
authors:
  - admin
tags:
  - Agent
  - LLM
# image:
#   caption: 'Image credit: [**Unsplash**](./featured.png)'
featured: true
---
> [Paper](https://arxiv.org/abs/2210.03629)

## Agent Work Setup

 - Observation，观察到的环境和可用信息
 - Action，可执行行为
 - Context，上下文窗口
  
    **Action Space**：Action Space 指 agent 能对外部环境执行、并会改变环境或得到反馈的动作集合，这些动作执行后，环境会返回 observation。

    **Language Space**：Language Space 指自由自然语言生成空间，在 ReAct 里，它不是“对外动作”，而是 agent 写给自己看的 thought / reasoning trace

## Motivation

当前的研究工作没有将agent的推理和行动结合起来完成目标任务，大都是仅用推理或者仅用行动来完成任务的。


ReAct提出一种一般范式，将LLM的推理和行动过程结合起来，以推理——行动——推理···这样互相交错的方式来达成目标，模型可以动态执行推理、调整行动规划和获取外部信息


## ReAct：Synergizing Reasoning & Acting

ReAct 的核心思想是：让大语言模型不要只“想”，也不要只“做”，而是把推理和行动交替结合起来。

传统方法大致分两类：

- CoT：模型只生成内部推理链，依赖自身知识。
- Act：模型只执行动作，比如搜索、点击、操作环境，但没有显式推理。

ReAct 把两者合在一起：
$$
Thought \rightarrow Action \rightarrow Observation \rightarrow Thought \rightarrow ...
$$
Thought：模型先用自然语言进行推理，比如分解目标、制定计划、提取关键信息、修正策略。
Action：模型执行外部动作，比如查询 Wikipedia、搜索、点击网页、操作游戏环境。
Observation：环境返回结果。
再 Thought：模型根据新观察继续推理，决定下一步。

### Comparing to other prompting methods
![Main tab](/ReAct/maintab.png)


![mode](/ReAct/success_fail_mode.png)

1. 表1中可以看出 ReAct+CoT 性能最好
   
   ReAct在HotpotQA和Fever两个benchmark上的性能均优于Act；在HotpotQA上弱于CoT，而在Fever上更优。这说明通过获取外部的正确信息来指导Agent行动能够切实提升Agent的任务性能；而CoT+ReAct的组合成功融合了两种范式的优点，取得了最好的性能。其作用机制是当第一种范式没有得到确定的答案时，用另一种范式来进行：
   
   （1）当CoT多数投票不够一致，说明模型内部知识可能不可靠，就切换到 ReAct，通过外部搜索/观察来补充事实，用ReAct为CoT兜底
   
   （2）如果 ReAct 在限定步数内没有得到答案，就退回 CoT-SC，用CoT为ReAct兜底
   
   从表1中可以看出，CoT-SC -> ReAct在两个benchmark上的性能均优于CoT-SC，说明在CoT推理得到的答案不确定时，通过ReAct获取外部信息能够有效指导Agent完成目标任务；而ReAct -> CoT-SC在两个benchmark上的性能也均优于ReAct，说明当Agent获取的外部信息对完成目标任务帮助微弱时，通过模型内部的自我推理能力来兜底

2. CoT范式的幻觉现象严重

    表2 展示了ReAct和CoT两个范式在HotpotQA上的成功/失败情景的样本统计对比，在失败场景中，CoT由幻觉导致的失败率高达56%

3. ReAcT范式虽然能够基于外部信息来推理答案，但是这样的思考过程限制了它的推理能力，没有CoT的推理能力强
    
    从表2 中可以看出，ReAct由于推理失败而导致的错误场景占比高达47%，但由幻觉导致的错误极少。
    > **为什么ReAct的推理能力不如CoT？**（From GPT）
    > 
    > 论文在 HotpotQA 的人工分析里发现，ReAct 的 reasoning error 是 47%，CoT 是 16%。主要原因有:
    > - ReAct 的推理结构约束. 
    > 
    >     CoT 可以连续展开一整段内部推理：`Question` $\rightarrow$ `Reasoning chian` $\rightarrow$ `Answer`, 而 ReAct 必须交替进行：`Thought` $\rightarrow$ `Action` $\rightarrow$ `Observasion` $\rightarrow$ `Thought`，这样的推理结构约束了ReAct模型的推理深度，在推理到一定程度后就要执行下一步的操作（Action/Observation），打断模型推理的过程。
    >
    > - ReAct依赖正确的外部信息
    >
    >     如果搜索结果为空、不相关或信息量不足，ReAct 的后续推理会被带偏。
    > 
    > - ReAct容易陷入重复的动作和思考循环
    >
    >     论文特别提到一种 ReAct 特有错误：模型重复生成之前的 thought/action，没能判断下一步应该换策略。这被归为 reasoning error。

4. 对于ReAcT来说，通过搜索来获取正确的信息丰富的知识十分重要

    由搜索错误而导致的错误率有23%，这在一定程度上反应出模型获取正确信息的重要性

> 为什么获取正确的外部信息很重要？没有这一步是不是就退化到CoT这样的范式？
>
> 获取正确的外部信息固然重要，ReAct需要依据外部的信息来调整未来的行动和规划。如果去掉外部信息获取和观察执行阶段，ReAct基本上就退化为CoT的推理范式。

![alt text](/ReAct/scaling_results.png)
在prompt推理的情况下，ReAct相比于其他范式在HotpotQA上没有性能优势，通常弱于其他范式，在模型参数量较小的情况下甚至不如Act-only和Standard，作者分析这可能是PALM-8/62B模型难以同时学习Acting和Reasoning；但在微调模型的情况下，ReAct范式取得了最佳的性能表现

### Decision making tasks

![alt text](/ReAct/decision_making_results.png)

除了通过prompting的方式使模型适应下游任务外，本文还对模型的决策能力进行了测试，在ALF World和Webshop两个benchmark上。

从表3中可以看出，ReAct范式取得了最好的性能，大幅超过Act和BUTLER基线；在表4中也取得了最好的成绩，说明在需要决策的场景下，模型一味的行动是不够的，通过思考、观察和接收外部消息来指导行动和决策十分重要，证明了ReAct内部推理机制的重要性

