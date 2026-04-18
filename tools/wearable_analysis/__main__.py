#!/usr/bin/env python3
"""
CLI entry point for wearable_analysis package.

Usage:
    python -m wearable_analysis --help
    python -m wearable_analysis portrait --data-dir path/to/data --sex female --age 30
    python -m wearable_analysis ingest --data-dir path/to/data
    python -m wearable_analysis discover --data-dir path/to/data
"""

import argparse
import sys
import os
import logging

# Add parent to path so `wearable_analysis` package is importable
# when running from arbitrary working directories
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('wearable_analysis')


def _resolve_paths(args):
    """Resolve data-dir and output-dir to absolute paths relative to project root."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tools_dir = os.path.dirname(script_dir)
    project_root = os.path.dirname(tools_dir)

    data_dir = args.data_dir
    if data_dir and not os.path.isabs(data_dir):
        data_dir = os.path.join(project_root, data_dir)

    output_dir = getattr(args, 'output_dir', None) or 'health_portrait_output'
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)

    return data_dir, output_dir, project_root


def cmd_portrait(args):
    """Run the full portrait pipeline: ingest -> discovery -> report."""
    from wearable_analysis.generate_portrait import main as portrait_main

    # Build sys.argv for generate_portrait's argparse
    sys_args = ['generate_portrait']
    sys_args += ['--data-dir', args.data_dir]
    if args.output_dir:
        sys_args += ['--output-dir', args.output_dir]
    sys_args += ['--sex', args.sex]
    sys_args += ['--age', str(args.age)]
    sys_args += ['--device', getattr(args, 'device', 'whoop')]
    if args.html_only:
        sys_args += ['--html-only']
    if getattr(args, 'no_open', False):
        sys_args += ['--no-open']
    sys_args += ['--lang', args.lang]

    # Temporarily replace sys.argv for generate_portrait's parser
    old_argv = sys.argv
    sys.argv = sys_args
    try:
        html_path = portrait_main()
    finally:
        sys.argv = old_argv

    # Send Telegram notification if requested
    if args.notify and html_path:
        _send_notification(html_path, args)


def cmd_ingest(args):
    """Run ingestion only — produce master.csv from raw device data."""
    from wearable_analysis.ingest import ingest_whoop, add_derived_features
    from wearable_analysis.adapters import get_adapter

    data_dir, output_dir, _ = _resolve_paths(args)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)

    device = getattr(args, 'device', 'whoop')

    if device == 'whoop':
        # Use existing ingest directly for WHOOP
        logger.info(f"Ingesting WHOOP data from {data_dir}...")
        result = ingest_whoop(data_dir, output_dir)
        df = result[0] if isinstance(result, tuple) else result
        df = add_derived_features(df)
    else:
        adapter = get_adapter(device)
        logger.info(f"Ingesting {device} data from {data_dir}...")
        df = adapter.ingest(data_dir, output_dir)

    csv_path = os.path.join(output_dir, 'data', 'master.csv')
    df.to_csv(csv_path)
    logger.info(f"Ingested {len(df)} days, {len(df.columns)} columns")
    logger.info(f"Saved: {csv_path}")
    return df


def cmd_discover(args):
    """Run correlation discovery on existing master.csv or fresh ingest."""
    import pandas as pd
    from wearable_analysis.discovery import run_discovery

    data_dir, output_dir, _ = _resolve_paths(args)

    # Try to load existing master.csv first
    csv_path = os.path.join(output_dir, 'data', 'master.csv')
    if os.path.exists(csv_path) and not args.reingest:
        logger.info(f"Loading existing master.csv from {csv_path}")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        logger.info("No master.csv found (or --reingest), running ingest first...")
        df = cmd_ingest(args)

    hypotheses_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hypotheses')
    logger.info("Running correlation discovery...")
    discovery = run_discovery(df, hypotheses_dir, output_dir)

    summary = discovery['summary']
    logger.info(
        f"Discovery complete: {summary['significant_after_fdr']} significant correlations, "
        f"{summary['coverage_pct']:.0f}% covered by hypotheses"
    )
    return discovery


def _send_notification(html_path, args):
    """Send Telegram notification with portrait summary."""
    import json as _json
    from pathlib import Path
    from urllib.request import Request, urlopen

    PERSONAL_OS_BOT_DIR = Path(os.getenv("TELEGRAM_BOT_DIR", Path.home() / ".telegram-bot"))

    # Load credentials
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
        logger.warning("No Telegram credentials found — notification skipped")
        return

    # Build summary message from grades
    try:
        from wearable_analysis.report import compute_domain_grades
        import pandas as pd
        output_dir = os.path.dirname(html_path)
        csv_path = os.path.join(output_dir, 'data', 'master.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            grades = compute_domain_grades(df, args.sex, args.age)
            grade_lines = "\n".join(
                f"  {k}: {v['grade']}" for k, v in grades.items()
            )
            text = (
                f"<b>Health Portrait Generated</b>\n\n"
                f"<b>Domain Grades:</b>\n<pre>{grade_lines}</pre>\n\n"
                f"Days: {len(df)} | Columns: {len(df.columns)}"
            )
        else:
            text = "<b>Health Portrait Generated</b>\nSee attached HTML report."
    except Exception as e:
        logger.warning(f"Could not compute grades for notification: {e}")
        text = "<b>Health Portrait Generated</b>\nSee attached HTML report."

    # Send text message
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
    if os.path.exists(html_path):
        import uuid
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


def main():
    parser = argparse.ArgumentParser(
        prog='wearable_analysis',
        description='Wearable Data Analysis — Literature-First N=1 Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python -m wearable_analysis portrait --data-dir path/to/whoop/data\n'
            '  python -m wearable_analysis ingest --data-dir path/to/data --device whoop\n'
            '  python -m wearable_analysis discover --data-dir path/to/data\n'
        ),
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # --- portrait ---
    p_portrait = subparsers.add_parser(
        'portrait',
        help='Full pipeline: ingest -> discovery -> population comparison -> HTML report',
    )
    p_portrait.add_argument('--data-dir', required=True,
                            help='Path to raw device data directory')
    p_portrait.add_argument('--output-dir', default='health_portrait_output',
                            help='Output directory for report and data files')
    p_portrait.add_argument('--sex', default='female',
                            help='Sex for population norms (male/female)')
    p_portrait.add_argument('--age', type=int, default=30,
                            help='Age for percentile calculations')
    p_portrait.add_argument('--device', default='whoop',
                            choices=['whoop', 'oura', 'garmin', 'csv'],
                            help='Device type (default: whoop)')
    p_portrait.add_argument('--html-only', action='store_true',
                            help='Skip hypothesis testing, generate EDA + discovery + report only')
    p_portrait.add_argument('--lang', default='ru', choices=['ru', 'en'],
                            help='Report language (default: ru)')
    p_portrait.add_argument('--notify', action='store_true',
                            help='Send Telegram notification with summary + HTML')
    p_portrait.add_argument('--no-open', action='store_true',
                            help='Do not open HTML report in browser')
    p_portrait.set_defaults(func=cmd_portrait)

    # --- ingest ---
    p_ingest = subparsers.add_parser(
        'ingest',
        help='Ingest raw device data into standardized master.csv',
    )
    p_ingest.add_argument('--data-dir', required=True,
                          help='Path to raw device data directory')
    p_ingest.add_argument('--output-dir', default='health_portrait_output',
                          help='Output directory')
    p_ingest.add_argument('--device', default='whoop',
                          choices=['whoop', 'oura', 'garmin', 'csv'],
                          help='Device type (default: whoop)')
    p_ingest.set_defaults(func=cmd_ingest)

    # --- discover ---
    p_discover = subparsers.add_parser(
        'discover',
        help='Run correlation discovery on ingested data',
    )
    p_discover.add_argument('--data-dir', required=True,
                            help='Path to raw device data directory')
    p_discover.add_argument('--output-dir', default='health_portrait_output',
                            help='Output directory')
    p_discover.add_argument('--device', default='whoop',
                            choices=['whoop', 'oura', 'garmin', 'csv'],
                            help='Device type (default: whoop)')
    p_discover.add_argument('--reingest', action='store_true',
                            help='Force re-ingest even if master.csv exists')
    p_discover.set_defaults(func=cmd_discover)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == '__main__':
    main()
