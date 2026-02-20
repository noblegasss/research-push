import argparse
import json
import os
from datetime import UTC, datetime

from app import (
    build_runtime_prefs_from_settings,
    build_daily_push_text,
    build_digest,
    fetch_candidates,
    format_email_body,
    get_backend_smtp_config,
    list_due_auto_push_subscriptions,
    mark_auto_push_run,
    post_webhook,
    send_email_digest,
)


def run_push_once(
    settings: dict,
    webhook_override: str = "",
    email_override: str = "",
    smtp_override: dict | None = None,
) -> tuple[bool, str]:
    prefs, _plan = build_runtime_prefs_from_settings(settings)
    papers, _fetch_note, _fetch_diag, effective_filter = fetch_candidates(prefs)
    prefs_runtime = dict(prefs)
    prefs_runtime["date_range_days"] = int(effective_filter.get("effective_days", prefs.get("date_range_days", 1)))
    prefs_runtime["strict_journal_only"] = bool(
        effective_filter.get("effective_strict_journal_only", prefs.get("strict_journal_only", True))
    )
    digest = build_digest(prefs_runtime, papers)
    push_text = build_daily_push_text(digest, lang=str(prefs_runtime.get("language", "zh")))

    webhook_url = webhook_override.strip()
    if not webhook_url:
        if bool(settings.get("enable_webhook_push", False)):
            webhook_url = str(settings.get("webhook_url", "")).strip()
    if not webhook_url:
        webhook_url = os.getenv("WEBHOOK_URL", "").strip()

    email_to = email_override.strip()
    if not email_to:
        if bool(settings.get("auto_send_email", False)):
            email_to = str(settings.get("email_to", "")).strip()
    if not email_to:
        email_to = os.getenv("EMAIL_TO", "").strip()

    backend_smtp_host, backend_smtp_port, backend_smtp_user, backend_smtp_password = get_backend_smtp_config()
    smtp_override = smtp_override or {}
    smtp_host = str(smtp_override.get("smtp_host", "") or backend_smtp_host or os.getenv("SMTP_HOST", ""))
    smtp_port = int(smtp_override.get("smtp_port", 0) or backend_smtp_port or os.getenv("SMTP_PORT", "587"))
    smtp_user = str(smtp_override.get("smtp_user", "") or backend_smtp_user or os.getenv("SMTP_USER", ""))
    smtp_password = str(
        smtp_override.get("smtp_password", "") or backend_smtp_password or os.getenv("SMTP_PASSWORD", "")
    )

    delivered = 0
    if webhook_url:
        ok, msg = post_webhook(
            webhook_url,
            {
                "date": datetime.now(UTC).strftime("%Y-%m-%d"),
                "today_new_summary": push_text["today_new_summary"],
                "worth_reading_summary": push_text["worth_reading_summary"],
                "digest": digest,
            },
            lang=str(prefs_runtime.get("language", "zh")),
        )
        if not ok:
            return False, msg
        print(msg)
        delivered += 1

    if email_to:
        ok, msg = send_email_digest(
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            smtp_user=smtp_user,
            smtp_password=smtp_password,
            to_email=email_to,
            subject=f"Research Digest {datetime.now(UTC).strftime('%Y-%m-%d')}",
            body=format_email_body(digest, push_text, lang=str(prefs_runtime.get("language", "zh"))),
            lang=str(prefs_runtime.get("language", "zh")),
        )
        if not ok:
            return False, msg
        print(msg)
        delivered += 1

    if delivered == 0:
        return False, "No delivery target. Provide webhook or email settings."
    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily paper digest push")
    parser.add_argument("--prefs", default="", help="Path to user preferences JSON")
    parser.add_argument("--all-due", action="store_true", help="Process all due multi-user auto-push subscriptions")
    parser.add_argument("--limit", type=int, default=0, help="Max subscriptions to process in --all-due mode")
    parser.add_argument("--webhook", default="", help="Webhook URL (fallback to WEBHOOK_URL env)")
    parser.add_argument("--email-to", default="", help="Recipient email (fallback to EMAIL_TO env)")
    parser.add_argument("--smtp-host", default="", help="SMTP host (fallback to SMTP_HOST env)")
    parser.add_argument("--smtp-port", type=int, default=587, help="SMTP port (fallback to SMTP_PORT env)")
    parser.add_argument("--smtp-user", default="", help="SMTP user (fallback to SMTP_USER env)")
    parser.add_argument("--smtp-password", default="", help="SMTP password/app password (fallback to SMTP_PASSWORD env)")
    args = parser.parse_args()

    smtp_override = {
        "smtp_host": args.smtp_host,
        "smtp_port": args.smtp_port,
        "smtp_user": args.smtp_user,
        "smtp_password": args.smtp_password,
    }

    if args.all_due:
        due = list_due_auto_push_subscriptions()
        if args.limit > 0:
            due = due[: args.limit]
        if not due:
            print("No due subscriptions.")
            return
        success = 0
        failed = 0
        for item in due:
            sid = item["subscriber_id"]
            ok, msg = run_push_once(settings=item["settings"], smtp_override=smtp_override)
            if ok:
                mark_auto_push_run(sid, item["local_date"], "")
                success += 1
                print(f"[ok] {sid[:8]} {item['timezone']} {item['local_time']}")
            else:
                mark_auto_push_run(sid, None, msg)
                failed += 1
                print(f"[failed] {sid[:8]}: {msg}")
        if failed:
            raise SystemExit(f"Completed with failures. success={success}, failed={failed}")
        print(f"Completed. success={success}, failed=0")
        return

    if not args.prefs:
        raise SystemExit("--prefs is required unless --all-due is used.")
    with open(args.prefs, "r", encoding="utf-8") as f:
        settings = json.load(f)
    ok, msg = run_push_once(
        settings=settings,
        webhook_override=args.webhook,
        email_override=args.email_to,
        smtp_override=smtp_override,
    )
    if not ok:
        raise SystemExit(msg)
    print(msg)


if __name__ == "__main__":
    main()
