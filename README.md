# Research Digest App

A Streamlit app that discovers new papers from selected journals/topics, de-duplicates results, enriches abstracts, and generates readable daily digests with optional AI-enhanced summaries.

Live app: https://research-push.streamlit.app/

## Features

- Journal-first discovery with multi-select support.
- Keyword/field filtering and exclude-keyword filtering.
- De-duplication by identifier (DOI/PMID/arXiv), with title/author/year fallback.
- Daily/weekly/custom time windows.
- Two primary content tabs:
  - Today Feed
  - Worth Reading (AI-selected when API is enabled)
- Journal-grouped paper feed with collapsible sections.
- Collapsible full abstract on each paper card.
- Optional AI summary enhancement (OpenAI API).
- Slack-compatible webhook push.
- Optional email delivery (frontend only asks for recipient email; SMTP is backend-managed).
- Bilingual UI support (Chinese/English).

## Project Structure

- `app.py`: Main Streamlit application.
- `daily_push.py`: CLI helper for scheduled push workflows.
- `requirements.txt`: Python dependencies.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## Settings and Persistence

The app supports safe per-user behavior for public deployments:

- Browser local settings cache is used for user preferences.
- Session state is used for generated digest display.
- Server-side persistence is disabled by default on Streamlit Cloud to avoid cross-user leakage.

Environment flags:

- `PUBLIC_MODE=1` to force public-safe mode.
- `SERVER_PERSISTENCE=1` only for trusted single-user/self-host setups.

## Data Sources and Fallbacks

Primary sources include Crossref, PubMed, RSS, and arXiv (depending on selected journals/queries).

Reliability behavior includes:

- short-term fetch cache,
- stale-cache fallback when upstream temporarily returns empty,
- journal-only mode for pure journal subscriptions,
- targeted multi-journal backfill when only one journal is initially hit.

## AI Configuration

To enable AI features:

1. Turn on **Use ChatGPT API** in Settings.
2. Provide a **Session API Key** in Settings (session-only in public-safe mode).
3. Choose model (default: `gpt-4.1-mini`).

If no API key is set, AI-specific output is hidden/fallback behavior is used.

## Slack Webhook Setup

### Create Incoming Webhook

1. Go to `https://api.slack.com/apps`
2. Create a new app.
3. Enable **Incoming Webhooks**.
4. Add webhook to a workspace channel.
5. Copy URL like:
   `https://hooks.slack.com/services/...`

### Configure in App

1. Open `⚙ Settings`
2. Enable **Webhook Push**
3. Paste Webhook URL
4. Save settings
5. Generate digest and click **Push to Webhook**

### What gets pushed

- Today's selected papers (with links)
- Worth Reading summary (with links when available)

## Email Delivery

Frontend:

- User provides only recipient email.

Backend SMTP (required):

- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USER`
- `SMTP_PASSWORD`

You can set these via Streamlit Secrets or environment variables.

## Deploy on Streamlit Community Cloud

1. Push code to GitHub.
2. Create app on Streamlit Cloud with:
   - repository: your repo
   - branch: `main`
   - main file: `app.py`
3. Add secrets/env vars if needed (OpenAI/SMTP).
4. Deploy and test.

## Daily Auto Push via GitHub Actions

This repo includes `.github/workflows/daily_slack_push.yml` to push digest to Slack daily.

Required setup:

1. Add `user_prefs.json` to repo root (the workflow reads this file).
2. In GitHub repo settings, add secret:
   - `SLACK_WEBHOOK_URL`
3. Enable Actions in the repository.
4. Optionally run once manually via **Actions -> Daily Slack Push -> Run workflow**.

Schedule:

- Default: `30 13 * * *` (13:30 UTC, about 08:30 US Eastern in standard time).

Security note:

- If a webhook URL was ever posted publicly, rotate/revoke it in Slack immediately and replace the GitHub secret.

## Multi-User Auto Push via Supabase Cron

Use this for per-user auto push on public Streamlit deployments.

1. Streamlit secrets:

```toml
AUTO_PUSH_DATABASE_URL="postgresql://...pooler...:6543/postgres?sslmode=require"
AUTO_PUSH_MULTIUSER="1"
SHOW_AUTO_PUSH_ADMIN_INFO="0"
```

2. GitHub repository secrets:

- `AUTO_PUSH_DATABASE_URL`
- Optional email: `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`
- Optional AI: `OPENAI_API_KEY`

3. Supabase SQL Editor:

- Run `supabase/sql/setup_auto_push_cron.sql`
- Replace `GITHUB_PAT` in that file first
- `GITHUB_PAT` should allow dispatching workflows on your repo (repo/actions write access)

4. Workflow trigger model:

- Supabase `pg_cron` runs every minute
- It calls GitHub `repository_dispatch` (`event_type: supabase-auto-push`)
- GitHub workflow `.github/workflows/auto_push_multiuser.yml` executes:

```bash
python daily_push.py --all-due
```
