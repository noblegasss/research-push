# Research Digest App 使用说明

在线地址：`https://research-push.streamlit.app/`

这个 App 用来按期刊/关键词抓取新论文，自动去重，并生成可读的每日摘要；你也可以把结果一键推送到 Slack。

## 1. 在 App 里怎么用

1. 打开 App，先进入 `⚙ Settings`。
2. 选择你关注的期刊（可多选）。
3. 填写关键词（可选），并设置排除关键词（例如 `survey`）。
4. 选择时间范围（如今天、最近 7 天或自定义）。
5. 点击生成/刷新后查看结果。
6. 在主页面查看：
   - `Today Feed`：今天/当前时间窗内的论文列表
   - `Worth Reading`：值得读（启用 AI 时会更完整）

## 2. （可选）启用 AI 总结

1. 进入 `⚙ Settings`。
2. 打开 **Use ChatGPT API**。
3. 填入 **Session API Key**。
4. 选择模型后保存。

未配置 API Key 时，App 仍可用，只是 AI 相关内容会降级或隐藏。

## 3. 在网页端使用 Slack 推送

### 3.1 先准备一个 Slack Webhook

1. 访问：`https://api.slack.com/apps`
2. 创建一个 Slack App，并开启 **Incoming Webhooks**。
3. 添加到目标频道后，复制 Webhook URL（形如 `https://hooks.slack.com/services/...`）。

### 3.2 在网页端接入 Webhook

1. 回到 App 的 `⚙ Settings`。
2. 打开 **Enable Webhook Push**。
3. 粘贴刚才的 Webhook URL。
4. 保存设置。
5. 生成摘要后，点击 **Push to Webhook** 推送到 Slack。

### 3.3 推送内容

- 当日/当前筛选条件下的论文条目（含链接）
- Worth Reading 摘要（有可用内容时）

## 4. 常见问题

- 没有抓到论文：先放宽时间范围，再减少筛选关键词。
- 点了推送但 Slack 没消息：检查 Webhook URL 是否完整、频道权限是否正确。
- 误泄露了 Webhook URL：立即在 Slack 里撤销并重新生成，再更新 App 设置。
