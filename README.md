# Research Digest App Guide

App URL: `https://research-push.streamlit.app/`

This app discovers new papers by journals/keywords, deduplicates results, and builds a readable daily digest. You can also push results to Slack.

## 1. How to Use the App

1. Open the app and go to `⚙ Settings`.
2. Select journals you want to follow (multi-select supported).
3. Add keywords (optional) and exclude keywords (for example, `survey`).
4. Choose a time range (today, last 7 days, or custom).
5. Generate/refresh to get results.
6. Review outputs in:
   - `Today Feed`: papers in the current time window
   - `Worth Reading`: prioritized picks (more complete when AI is enabled)

## 2. (Optional) Enable AI Summary

1. Go to `⚙ Settings`.
2. Turn on **Use ChatGPT API**.
3. Enter your **Session API Key**.
4. Choose a model and save.

Without an API key, the app still works, but AI-specific parts are hidden or downgraded.

## 3. Use Slack Push in Web App

### 3.1 Prepare a Slack Webhook

1. Visit `https://api.slack.com/apps`
2. Create a Slack app and enable **Incoming Webhooks**.
3. Add it to a target channel, then copy the webhook URL (`https://hooks.slack.com/services/...`).

### 3.2 Connect Webhook in the App

1. Go back to `⚙ Settings` in the app.
2. Turn on **Enable Webhook Push**.
3. Paste your webhook URL.
4. Save settings.
5. After generating a digest, click **Push to Webhook**.

### 3.3 What Gets Pushed

- Paper items in your current digest (with links)
- Worth Reading summary (when available)

## 4. Troubleshooting

- No papers found: widen the time range and reduce keyword filters.
- Slack received nothing: check webhook URL validity and channel permissions.
- Webhook leaked: revoke/rotate it in Slack immediately, then update the app.
