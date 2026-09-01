# 上传到 GitHub

本仓库已经初始化为 Git 仓库，并在 `.gitattributes` 中配置了 Git LFS。
首次上传前，请先在 GitHub 创建一个空仓库，不要勾选 README、`.gitignore` 或 License。

在本目录执行：

```powershell
git status
git remote add origin https://github.com/<你的用户名>/battery-rul-nasa5-basilisk-v10.git
git push -u origin main
```

如果使用 SSH，将远程地址替换为：

```powershell
git@github.com:<你的用户名>/battery-rul-nasa5-basilisk-v10.git
```

首次推送会同时上传两个 Git LFS 数据文件，约 193 MB。需要确保 GitHub 账户已启用 Git LFS 配额。若只需发布代码和权重，可在推送前运行 `git lfs ls-files` 确认被 LFS 跟踪的文件。
