# Graduation-Design
this is a repository where conclude the code of Zjb's graduation design: The recognition algorithm based on model compression

# 将本地仓库push到github
1. 初始化本地仓库  
`git init`

2. 在本地仓库跟目录添加.gitignore文件  
如果要保留文件目录，请在要保留的目录下  
添加  **.gitignore** 内容如下：
```
*
!.gitignore
```

3. 测试github连接  
`ssh -T git@github.com`  
若输出为下面的则连接成功  
`Hi zip95297! You've successfully authenticated, but GitHub does not provide shell access.`

4. 在本地仓库中连接github仓库  
`git remote add origin git@github.com:zip95297/Graduation-Design.git`  

5. 注意：由于新建github仓库时可能新建了readme.md  
先使用rebase方法合并branch`git pull --rebase origin main`

6. 添加到stage  
`git add .`  

7. 提交到本地仓库   
`git commit -m "message to asmbel the change"`  

8. 提交到远程仓库，由于初始远程仓库为空，使用-u参数  
` git push -u origin main`  
之后直接 **#below#** 即可  
`git push origin main`

9. TIPS  
`rm -rf .git`删除本地git仓库无需确认  
`git diff`快速查看当前和已经commit的区别  
`git status`查看当前仓库的状态（哪些文件没有add、commit）  
`git log`查看本地仓库的版本  
`git rm --cache file/path`删除暂存文件  
!!!没有cache就删掉了
