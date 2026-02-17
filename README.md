# Research Digest App

Streamlit 应用：按用户偏好抓取候选论文，去重、筛选、评分，并输出可读版论文推荐（介绍/方法总结/价值判断）。

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## 功能

- 用户自定义 `keywords`、`journals`（多选+自定义）、`exclude_keywords`、时间窗口和数量上限
- 支持自动抓取（arXiv + Crossref），聚焦“当天新论文”
- 支持可选 `Institution Proxy Prefix`（学校代理前缀）以便跳转受限期刊链接
- 支持可选 ChatGPT API 增强中文总结（介绍/方法/价值）
- 支持 Webhook 推送（Slack Incoming Webhook 兼容）
- 支持 SMTP 邮件推送（前端仅需收件邮箱；SMTP 由后台统一配置）
- 提供 `daily_push.py` 可配合 cron 做每日自动推送
- 去重规则：
  - 优先 DOI / PMID / arXiv ID
  - 其次 `title + first_author + year`
- 评分维度：`relevance` / `novelty` / `rigor` / `impact`，并按权重生成 `total`
- 页面默认输出可读卡片；同时可展开查看/下载结构化 JSON

## 注意

- 当 `abstract` 缺失时，卡片会自动标注：
  - `Abstract unavailable; summary is tentative.`
- 自动抓取依赖外网访问 API（arXiv/Crossref）。
- 学校账号登录流程在浏览器内完成，应用本身不保存学校账号密码。

## 推送到 Slack（详细）

### 1. 在 Slack 创建 Incoming Webhook

1. 打开 `https://api.slack.com/apps`
2. 创建一个 App（From scratch）
3. 进入 `Incoming Webhooks`，开启开关
4. 点击 `Add New Webhook to Workspace`
5. 选择你要接收消息的频道
6. 复制生成的 Webhook URL（形如 `https://hooks.slack.com/services/...`）

### 2. 在 App 中配置

1. 打开应用里的 `⚙ 设置`
2. 打开 `启用 Webhook 推送`
3. 粘贴 `Webhook URL`
4. 点击 `保存设置`
5. 生成 Digest 后，点击 `推送到 Webhook`

### 3. 你会收到什么

- `今日论文（含链接）`：当天抓到的全部入选论文 + 链接
- `值得读`：AI 选出的优先阅读论文摘要（若已启用并配置 API）

### 4. 常见问题排查

- 提示 `Webhook URL 为空`
  - 设置里未保存 URL，或本地缓存被清空后未重新保存
- 点击推送无消息
  - 检查 Webhook URL 是否以 `https://hooks.slack.com/services/` 开头
  - 检查频道是否还存在、机器人是否仍有发言权限
- HTTP 非 2xx
  - 通常是 URL 无效、被撤销、或 workspace 权限策略拦截
- 生成有论文但 Slack 没有“值得读”
  - 未启用 ChatGPT API、未填 API key，或模型请求失败/超时

## 每日自动推送（cron）

1. 先在 App 里下载一份 `user_prefs.json`
2. 用命令测试一次：

```bash
python daily_push.py --prefs user_prefs.json --webhook "https://your-webhook-url"
```

3. 发邮件测试（Gmail 示例）：

```bash
python daily_push.py \
  --prefs user_prefs.json \
  --email-to "your_email@example.com" \
  --smtp-host "smtp.gmail.com" \
  --smtp-port 587 \
  --smtp-user "your_gmail@gmail.com" \
  --smtp-password "your_app_password"
```

说明：Gmail 需要使用 App Password，不是登录密码。

4. 配置 cron（示例：每天早上 8:30）：

```bash
30 8 * * * cd "/path/to/research_push" && /path/to/research_push/.venv/bin/python daily_push.py --prefs user_prefs.json --webhook "https://your-webhook-url" >> /tmp/research_digest.log 2>&1
```
