# 在 GitHub Desktop 中添加本包

本版 ZIP 中的 `PertRank` 文件夹已初始化为本地 Git 仓库，尚未提交或连接远程仓库。仓库根目录就是同时包含 `README.md`、`reproduce.py`、`modules` 和 `docs` 的这一层。

1. 将本版 ZIP 解压到较短的位置，例如 `D:\GitHub\`，得到 `D:\GitHub\PertRank\`。保留 ZIP 中的 `.git` 文件夹。
2. 打开 GitHub Desktop，选择 **File → Add Local Repository…**。
3. 点击 **Choose…**，选择 `D:\GitHub\PertRank`，然后 **Add Repository**。不要选择 `modules`，也不要在这里再次创建名叫 PertRank 的子文件夹。
4. **Changes** 应显示本包的代码和数据文件。填写 Summary，例如 `Import PertRank reproducibility package v1.5.4`，点击 **Commit to main**。
5. 点击 **Publish repository**。按本项目匿名评审要求，保留 **Keep this code private** 勾选，再发布到自己的账户。

可以用 **Repository → Show in Explorer** 检查位置：打开后应直接看到 `README.md`、`modules` 和 `docs`。如果只看到一个空文件夹，说明 GitHub Desktop 打开的是错误层级。

此前的截图中，`PertRank` 子文件夹与 `modules`、`docs` 并列。这提示 New Repository 可能在代码目录内部另建了仓库。GitHub Desktop 的 New Repository 会把 Name 追加到 Local path 下，因此把现有代码目录直接填入 Local path，可能使代码落在新仓库之外。截图未提供报错对话框，无法据此判断账户、网络或发布权限是否还有问题。

本版修复了包内过长的存档路径、版本标识及跨平台换行保护。原始路径对应关系在 `WINDOWS_PATH_MAP.csv` 中。`.gitattributes` 保留科学文件的原始字节，避免 Git 的换行转换破坏 SHA-256 校验。请保留该文件。

如果添加时仍报“不是 Git 仓库”，先确认 `.git` 已被完整解压，并确认选中的是本版 PertRank 根目录。如果发布失败，请提供 GitHub Desktop 的具体错误窗口；文件列表截图不能说明登录或网络失败原因。

官方操作说明：https://docs.github.com/en/desktop/overview/creating-your-first-repository-using-github-desktop
