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
        - git rm --cached `file` 将已添加到暂存区的文件撤回
        - git add . 将当前文件夹下的所有文件都添加到暂存区中
    - git commit 提交（只会提交**暂存区**中的文件）
        - 参数"-m"：添加提交备注
2. git clone  从服务器上拷贝仓库
3. git log  查看当前仓库提交记录
    - 参数“--oneline” 显示简略的提交记录
4. git ls-files  查看暂存区文件
5. git reflog  查看历史记录
6. git diff  查看工作区、暂存区、版本仓库之间的差异
    - git diff  比较当前工作区和暂存区之间的差异
    - git diff HEAD 比较工作区和版本仓库之间的差异
    - git diff --cached 比较工作区和暂存区的差异
    - git diff 版本ID1 版本ID2 比较两个提交版本之间的差异,版本2相较于版本1做出的更改
    > HEAD 指向分支的最新提交版本，HEAD～或HEAD^表示上一个提交版本。HEAD～2表示HEAD之前的两个版本
7. git rm `文件名`  同时在工作区和暂存区中删除文件
    - git rm -r *  递归删除某个目录下的所有子目录和文件，删除后记得要提交
    - git rm --cached `文件名`  把文件从暂存区中删除，但是在工作区中仍然保存
### Git本地数据管理

1. 工作区
    本地文件目录 ".git/"
2. 暂存区 Staging Area/Index
    临时存储区，用于保存即将提交到Git仓库的修改内容 ".git/index/"
3. 本地仓库
    使用git init/clone创建的仓库 .git/objects/；是Git存储代码和版本信息的主要位置
![Img](/Git/git_procedure.png)
    
### Git中的文件状态

1. 未跟踪: 刚刚创建还未被git管理
2. 未修改: 已经由git管理但是文件的内容还未变化
3. 已修改: 已经修改的文件，但是尚未被添加到暂存区
4. 已暂存: 已经修改的文件已添加到暂存区

### Git Reset 回退
- git reset --soft
    - 回退到某一个版本，并保存工作区和暂存区的内容
- git reset --hard
    - 回退到某一个版本，并丢弃工作区和暂存区的内容
- git reset --mixed （默认）
    - 回退到某一个版本，只保留工作区的内容

### .gitignore文件
版本库中不应保留的文件：
1. 系统或者软件自动生成的文件
2. 编译产生的中间件和结果文件
3. 运行时生成的日志文件、缓存文件、临时文件等
4. 涉及身份、密码、口令、密钥等敏感信息的文件
- 将文件名写入.gitignore文件中即可忽略提交（目录要在之后加‘/’）
忽略前提：这个文件不能是已经被加入到仓库中的文件
空文件夹也不会被加入到仓库中
- .gitignore文件匹配规则
    - 从上到下逐行匹配，每一行表示一个忽略模式
    - 空行或者以‘#’开头的行会被Git忽略，#用于注释
    - 模糊匹配规则
- 示例
    1. 只忽略当前目录下的TODO文件：/TODO
    2. 忽略任何目录下名为build的文件夹：build/
常用语言忽略模版：[link](https://github.com/github/gitignore)

### 远程仓库
1. git pull `remote`  拉取远程仓库，同时同步本地和远程仓库的修改。git fetch只获取远程仓库的修改内容。
2. git push `remote` `branch`  推送本地仓库
    git push -u origin main:main 将本地仓库与远程仓库origin关联，将本地仓库的main分支推送到远程仓库的main分支
3. git remote add `别名` `url`  添加一个远程仓库
4. git remote -v 查看远程仓库信息