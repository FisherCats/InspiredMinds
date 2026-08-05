---
title: Hello Agent 学习笔记 / 01 Agent基础
slug: hello-agent-01
date: 2026-08-01
summary: 认识智能体的演化、分类与 Agent Loop，并比较 Agent 和传统 Workflow 的工作方式。
authors:
  - admin
tags:
  - Agent
  - LLM
---

> [Lecture 1](https://hello-agents.datawhale.cc/#/./chapter1/%E7%AC%AC%E4%B8%80%E7%AB%A0%20%E5%88%9D%E8%AF%86%E6%99%BA%E8%83%BD%E4%BD%93)

本节介绍了什么是Agent？Agent的演化和分类、现代Agent的运行机制、Agent实操以及现代Agent工作与传统Workflow的差异

## Agent运行规则
![现代智能体的 Agent Loop](/tutorials-assets/01-agent基础/image.png)
上图展示了现代智能体运行的核心机制，这个核心机制被称为**智能体循环（Agent Loop）**。首先从Perception开始，智能体感知周围的环境信息，随后开始思考推理，规划接下来应当怎么做。决策完成后，智能体通过执行器完成规划的动作。如此反复，智能体不断的思考，与外界环境交互获取信息，给出最终的答案。

## Agent和传统Work flow的工作差异
与工作流不同，基于大型语言模型的智能体是一个具备自主性的、以目标为导向的系统。它不仅仅是执行预设指令，而是能够在一定程度上理解环境、进行推理、制定计划，并动态地采取行动以达成最终目标。
![Agent 与传统 Workflow 的差异](/tutorials-assets/01-agent基础/image-1.png)
简单来说，Workflow 是让 AI 按部就班地执行指令，而 Agent 则是赋予 AI 自由度去自主达成目标
