import argparse
import json
import os
from datetime import UTC, datetime

from app import (
    build_runtime_prefs_from_settings,
    build_daily_push_text,
    build_digest,
    fetch_candidates,
    list_due_auto_push_subscriptions,
    mark_auto_push_run,
    post_webhook,
)


def run_push_once(
    settings: dict,
    webhook_override: str = "",
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

    if delivered == 0:
        return False, "No delivery target. Provide webhook settings."
    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily paper digest push")
    parser.add_argument("--prefs", default="", help="Path to user preferences JSON")
    parser.add_argument("--all-due", action="store_true", help="Process all due multi-user auto-push subscriptions")
    parser.add_argument("--limit", type=int, default=0, help="Max subscriptions to process in --all-due mode")
    parser.add_argument("--webhook", default="", help="Webhook URL (fallback to WEBHOOK_URL env)")
    args = parser.parse_args()

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
            ok, msg = run_push_once(settings=item["settings"])
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
    )
    if not ok:
        raise SystemExit(msg)
    print(msg)


if __name__ == "__main__":
    main()
