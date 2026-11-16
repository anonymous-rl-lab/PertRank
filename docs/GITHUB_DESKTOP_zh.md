# 在 GitHub Desktop 中使用审稿包

本版只打包代码与数据，不包含 `.git` 元数据。仓库根目录是同时包含 `README.md`、`reproduce.py`、`modules` 和 `docs` 的 `PertRank` 文件夹。

## 更新已有仓库

1. 在 GitHub Desktop 中选择已有 PertRank 仓库，并用 **Repository → Show in Explorer** 确认根目录。
2. 保留该目录内的 `.git`，将新包中的工作文件复制到该根目录。不要把整个 `PertRank` 再套进一个同名子目录。
3. 检查 Changes，提交新版本说明，再按作者选择的发布方式推送。包本身不会推送或修改远程仓库。

## 创建新仓库

在解压后的 `PertRank` 根目录打开终端执行 `git init -b main`，然后在 GitHub Desktop 中选择 **File → Add Local Repository**，选择这一根目录。另一种方式是先由 GitHub Desktop 创建空仓库，再把本包内容复制进新仓库的根目录。避免在已有代码目录内误建一层同名子目录。

请保留 `.gitattributes`，它用于防止换行转换改变科学文件字节及 SHA-256。若需要匿名评审，公开前由作者检查整个仓库、历史记录和访问链接是否符合匿名要求。
