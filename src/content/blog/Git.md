---
title: Git Technic
summary: Threads of Git 
date: 2024-12-17
authors:
  - admin
tags:
  - git
  - Remote Repo
# image:
#   caption: 'Image credit: [**Unsplash**](./featured.png)'
featured: true
---
# Git Notes

## Git 常用命令
### 配置用户
```bash
 git config --global/local user.name "FisherCat"
 
 git config --global/local user.email fishercat_@outlook.com
```
global/local表示的是对全局仓库有效还是仅对当前本地仓库有效
```bash
 git config --global credential.helper store
 保存信息下次登录时无需再次输入
 
  git config --global --list
```
查看配置信息

### 创建仓库

1. git init  从本地创建仓库
    - git status 查看仓库状态
    - git add  添加到暂存区
        - git rm --cached *file* 将已添加到暂存区的文件撤回
        - git add . 将当前文件夹下的所有文件都添加到暂存区中
    - git commit 提交（只会提交**暂存区**中的文件）
        - 参数"-m"：添加提交备注
