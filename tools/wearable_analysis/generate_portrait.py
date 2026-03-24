#!/usr/bin/env python3
"""
Generate N=1 Health Portrait from wearable data.

Orchestrates the full pipeline: ingest -> discovery -> personalization -> HTML report.

Usage:
    python3 tools/wearable_analysis/generate_portrait.py \\
        --data-dir "path/to/whoop/data" \\
        --output-dir "output/reports" \\
        --sex female --age 30

    # Quick mode: skip hypothesis testing, just EDA + discovery + report
    python3 tools/wearable_analysis/generate_portrait.py \\
        --data-dir "path/to/data" --html-only
"""

import argparse
import sys
import os
import logging

# Add parent to path so `wearable_analysis` package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('portrait')


def main():
    parser = argparse.ArgumentParser(
        description='Generate N=1 Health Portrait from wearable data')
    parser.add_argument('--data-dir', required=True,
                        help='Path to WHOOP JSON data directory')
    parser.add_argument('--output-dir', default='health_portrait_output',
                        help='Output directory for report and data files')
    parser.add_argument('--sex', default='female',
                        help='Sex for population norms (male/female)')
    parser.add_argument('--age', type=int, default=30,
                        help='Age for percentile calculations')
    parser.add_argument('--html-only', action='store_true',
                        help='Skip hypothesis testing, generate EDA + discovery + report only')
    parser.add_argument('--lang', default='ru', choices=['ru', 'en'],
                        help='Report language (default: ru)')
    parser.add_argument('--notify', action='store_true',
                        help='Send Telegram notification with summary + HTML report')
    parser.add_argument('--no-open', action='store_true',
                        help='Do not open HTML report in browser')
    args = parser.parse_args()

    # Import pipeline modules (after path setup)
    from wearable_analysis.ingest import ingest_whoop, add_derived_features
    from wearable_analysis.discovery import run_discovery
    from wearable_analysis.report import compute_domain_grades, generate_html_report
    from wearable_analysis.personalize import population_comparison
    from wearable_analysis.config import load_user_config

    # ── Auto-setup: create user_config.yaml if missing ───────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'user_config.yaml')
    template_path = os.path.join(script_dir, 'user_config.template.yaml')
    if not os.path.exists(config_path) and os.path.exists(template_path):
        import shutil
        shutil.copy2(template_path, config_path)
        logger.info(f"Created user_config.yaml from template — edit it to customize")

    # ── Auto-setup: create venv if missing ────────────────────────
    venv_path = os.path.join(script_dir, '.venv')
    if not os.path.exists(venv_path):
        logger.info("No .venv found — hint: run 'python3 -m venv .venv && source .venv/bin/activate && pip install pandas numpy scipy scikit-learn matplotlib seaborn pyyaml statsmodels'")

    # Resolve paths relative to project root (grandparent: tools/wearable_analysis/ → tools/ → project_root)
    # script_dir already set above
    tools_dir = os.path.dirname(script_dir)                   # tools/
    project_root = os.path.dirname(tools_dir)                 # project root
    if os.path.isabs(args.data_dir):
        data_dir = args.data_dir
    else:
        data_dir = os.path.join(project_root, args.data_dir)

    if os.path.isabs(args.output_dir):
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(project_root, args.output_dir)

    hypotheses_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'hypotheses')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)

    # ── Step 1: Ingest ──────────────────────────────────────────────
    logger.info(f"Ingesting WHOOP data from {data_dir}...")
    result = ingest_whoop(data_dir, output_dir)
    # ingest_whoop returns (master, activities, healthspan) tuple
    df = result[0] if isinstance(result, tuple) else result
    df = add_derived_features(df)
    logger.info(f"Ingested {len(df)} days, {len(df.columns)} columns")

    # Save master CSV
    csv_path = os.path.join(output_dir, 'data', 'master.csv')
    df.to_csv(csv_path)
    logger.info(f"Saved master.csv: {csv_path}")

    # ── Step 2: Discovery (all-pairs correlation) ───────────────────
    logger.info("Running correlation discovery...")
    discovery = run_discovery(df, hypotheses_dir, output_dir)
    logger.info(
        f"Discovery: {discovery['summary']['significant_after_fdr']} significant, "
        f"{discovery['summary']['coverage_pct']:.0f}% covered"
    )

    # ── Step 3: Population comparison ───────────────────────────────
    logger.info("Computing population comparison...")
    population = population_comparison(df, args.sex, args.age)

    # ── Step 4: Domain grades ───────────────────────────────────────
    logger.info("Computing domain grades...")
    grades = compute_domain_grades(df, args.sex, args.age)

    # ── Step 5: Generate HTML report ────────────────────────────────
    html_path = os.path.join(output_dir, 'health_portrait.html')
    logger.info(f"Generating HTML report: {html_path}")

    user_config = load_user_config()
    generate_html_report(
        df=df,
        grades=grades,
        discovery=discovery,
        hypothesis_results=None,  # TODO: integrate hypothesis testing
        population=population,
        actions=None,  # TODO: integrate actionability scoring
        output_path=html_path,
        user_config=user_config,
        lang=args.lang,
    )

    # ── Step 6: Copy to Health Portal ─────────────────────────────
    portal_path = os.path.join(
        project_root, 'web', 'health-portal', 'public', 'health_portrait.html')
    portal_dir = os.path.dirname(portal_path)
    if os.path.isdir(portal_dir):
        import shutil
        shutil.copy2(html_path, portal_path)
        logger.info(f"Copied to Health Portal: {portal_path}")
    else:
        logger.info(f"Health Portal dir not found ({portal_dir}), skipping copy")

    # ── Summary ─────────────────────────────────────────────────────
    logger.info(f"Health Portrait generated: {html_path}")
    grade_summary = ', '.join(
        f'{k}={v["grade"]}' for k, v in grades.items())
    logger.info(f"Domain grades: {grade_summary}")
    logger.info(f"Output directory: {output_dir}")

    # ── Open in browser ───────────────────────────────────────────
    if not getattr(args, 'no_open', False):
        import webbrowser
        url = 'file://' + os.path.abspath(html_path)
        webbrowser.open(url)
        logger.info(f"Opened in browser: {url}")

    # ── Optional: Telegram notification ───────────────────────────
    if getattr(args, 'notify', False):
        _notify_telegram(html_path, grades, df, args)

    return html_path


def _notify_telegram(html_path, grades, df, args):
    """Send portrait summary + HTML file to Telegram."""
    import json as _json
    from pathlib import Path
    from urllib.request import Request, urlopen
    import uuid

    PERSONAL_OS_BOT_DIR = Path.home() / "Cursor" / "your-telegram-bot"

    # Load credentials (same pattern as tools/notify_research.py)
    bot_token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        env_file = PERSONAL_OS_BOT_DIR / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("TELEGRAM_TOKEN=") and not bot_token:
                    bot_token = line.split("=", 1)[1].strip().strip("\"'")
                elif line.startswith("ALLOWED_USERS=") and not chat_id:
                    users = line.split("=", 1)[1].strip().strip("\"'")
                    chat_id = users.split(",")[0].strip()

    if not bot_token or not chat_id:
        logger.warning("No Telegram credentials — notification skipped")
        return

    # Build summary message
    grade_lines = "\n".join(
        f"  {k}: {v['grade']}" for k, v in grades.items()
    )
    text = (
        f"<b>Health Portrait Generated</b>\n\n"
        f"<b>Domain Grades:</b>\n<pre>{grade_lines}</pre>\n\n"
        f"Days: {len(df)} | Columns: {len(df.columns)}"
    )

    payload = _json.dumps({
        "chat_id": int(chat_id),
        "text": text,
        "parse_mode": "HTML",
    }).encode("utf-8")

    req = Request(
        f"https://api.telegram.org/bot{bot_token}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urlopen(req, timeout=15)
        if resp.status == 200:
            logger.info("Telegram summary sent")
    except Exception as e:
        logger.warning(f"Telegram message failed: {e}")

    # Send HTML file as document
    if not os.path.exists(html_path):
        return

    boundary = uuid.uuid4().hex
    parts = []
    parts.append(
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="chat_id"\r\n\r\n{chat_id}'
    )
    parts.append(
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="caption"\r\n\r\n'
        f'N=1 Health Portrait'
    )

    body_prefix = ("\r\n".join(parts) + "\r\n").encode("utf-8")
    filename = os.path.basename(html_path)
    doc_header = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="document"; filename="{filename}"\r\n'
        f'Content-Type: text/html\r\n\r\n'
    ).encode("utf-8")

    with open(html_path, "rb") as f:
        doc_data = f.read()

    body_suffix = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = body_prefix + doc_header + doc_data + body_suffix

    req = Request(
        f"https://api.telegram.org/bot{bot_token}/sendDocument",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        resp = urlopen(req, timeout=30)
        if resp.status == 200:
            logger.info("HTML report sent to Telegram")
    except Exception as e:
        logger.warning(f"Telegram document send failed: {e}")


if __name__ == '__main__':
    main()
