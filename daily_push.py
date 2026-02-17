import argparse
import json
import os
from datetime import UTC, datetime

from app import (
    build_daily_push_text,
    build_digest,
    fetch_candidates,
    format_email_body,
    post_webhook,
    send_email_digest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily paper digest push")
    parser.add_argument("--prefs", required=True, help="Path to user preferences JSON")
    parser.add_argument("--webhook", default="", help="Webhook URL (fallback to WEBHOOK_URL env)")
    parser.add_argument("--email-to", default="", help="Recipient email (fallback to EMAIL_TO env)")
    parser.add_argument("--smtp-host", default="", help="SMTP host (fallback to SMTP_HOST env)")
    parser.add_argument("--smtp-port", type=int, default=587, help="SMTP port (fallback to SMTP_PORT env)")
    parser.add_argument("--smtp-user", default="", help="SMTP user (fallback to SMTP_USER env)")
    parser.add_argument("--smtp-password", default="", help="SMTP password/app password (fallback to SMTP_PASSWORD env)")
    args = parser.parse_args()

    with open(args.prefs, "r", encoding="utf-8") as f:
        prefs = json.load(f)

    prefs["date_range_days"] = 1
    papers = fetch_candidates(prefs)
    digest = build_digest(prefs, papers)
    push_text = build_daily_push_text(digest)

    webhook_url = args.webhook or os.getenv("WEBHOOK_URL", "")
    if webhook_url.strip():
        ok, msg = post_webhook(
            webhook_url,
            {
                "date": datetime.now(UTC).strftime("%Y-%m-%d"),
                "today_new_summary": push_text["today_new_summary"],
                "worth_reading_summary": push_text["worth_reading_summary"],
                "digest": digest,
            },
        )
        if not ok:
            raise SystemExit(msg)
        print(msg)

    email_to = args.email_to or os.getenv("EMAIL_TO", "")
    smtp_host = args.smtp_host or os.getenv("SMTP_HOST", "")
    smtp_port = int(os.getenv("SMTP_PORT", str(args.smtp_port)))
    smtp_user = args.smtp_user or os.getenv("SMTP_USER", "")
    smtp_password = args.smtp_password or os.getenv("SMTP_PASSWORD", "")
    if email_to.strip():
        ok, msg = send_email_digest(
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            smtp_user=smtp_user,
            smtp_password=smtp_password,
            to_email=email_to,
            subject=f"Research Digest {datetime.now(UTC).strftime('%Y-%m-%d')}",
            body=format_email_body(digest, push_text),
        )
        if not ok:
            raise SystemExit(msg)
        print(msg)

    if not webhook_url.strip() and not email_to.strip():
        raise SystemExit("No delivery target. Provide webhook or email settings.")


if __name__ == "__main__":
    main()
