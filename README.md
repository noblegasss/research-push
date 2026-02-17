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
- 支持 Webhook 推送“今日新论文总结 + 值得读总结”
- 支持 SMTP 邮件推送（可直接发到个人邮箱）
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
30 8 * * * cd "/Users/weizhang/Dropbox (Personal)/research_push" && /Users/weizhang/Dropbox\ (Personal)/research_push/.venv/bin/python daily_push.py --prefs user_prefs.json --webhook "https://your-webhook-url" >> /tmp/research_digest.log 2>&1
```
