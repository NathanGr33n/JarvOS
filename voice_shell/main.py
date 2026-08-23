import argparse
import asyncio
import logging
from pathlib import Path
from typing import List, Optional

from voice_shell.src.config import Config


def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JarvOS Voice Shell")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to YAML configuration file (default: ./config.yaml).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )

    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run", help="Start the voice shell orchestrator (default).")

    status_parser = sub.add_parser(
        "status",
        help="Report engine health-gate status and supervised service snapshot.",
    )
    status_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )

    models_parser = sub.add_parser(
        "models",
        help="Manage local pre-quantized model downloads (app store).",
    )
    models_parser.add_argument(
        "--dir",
        type=Path,
        default=Path("models"),
        help="Models root directory (default: ./models).",
    )
    models_sub = models_parser.add_subparsers(dest="models_command", required=True)
    models_sub.add_parser("list", help="List catalog entries.")
    models_sub.add_parser("status", help="Show local download/verification status.")
    download_parser = models_sub.add_parser("download", help="Download a model by name.")
    download_parser.add_argument("name", help="Catalog entry name (see 'models list').")
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a verified copy already exists.",
    )

    args = parser.parse_args(argv)
    if args.command is None:
        args.command = "run"
    return args


async def _run_orchestrator(config_path: Path) -> None:
    from voice_shell.src.orchestrator import Orchestrator

    config = Config.from_yaml(config_path) if config_path.exists() else Config()
    orchestrator = Orchestrator(config=config)
    await orchestrator.run()


async def _run_status(config_path: Path, fmt: str) -> int:
    from voice_shell.src.diagnostics import collect_diagnostic_report, format_report

    path = config_path if config_path is not None and config_path.exists() else None
    if config_path is not None and not config_path.exists():
        logging.getLogger(__name__).warning("Config not found at %s; using defaults.", config_path)
    report = await collect_diagnostic_report(path)
    print(format_report(report, fmt=fmt))
    if not report.health_enabled:
        return 0
    return 0 if report.ready else 1


def _run_models(args: argparse.Namespace) -> int:
    from voice_shell.src.appstore import DownloadError, download, list_models, status_all

    models_dir: Path = args.dir

    if args.models_command == "list":
        for entry in list_models():
            size = ""
            if entry.size_bytes:
                gib = entry.size_bytes / (1024 ** 3)
                size = f" (~{gib:.1f} GiB)" if gib >= 1 else f" (~{entry.size_bytes / (1024 ** 2):.0f} MiB)"
            print(f"{entry.name} [{entry.category}] - {entry.description}{size}")
        return 0

    if args.models_command == "status":
        for status in status_all(models_dir):
            state = "verified" if status.verified else ("downloaded" if status.downloaded else "missing")
            detail = f" ({status.detail})" if status.detail else ""
            print(f"{status.entry.name}: {state}{detail}")
        return 0

    if args.models_command == "download":
        try:
            result = download(args.name, models_dir, force=args.force)
        except DownloadError as exc:
            print(f"Error: {exc}")
            return 1
        print(f"Downloaded {result.entry.name} -> {result.path}")
        return 0

    return 1


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    _configure_logging(args.log_level)
    config_path = args.config if isinstance(args.config, Path) else Path("config.yaml")

    try:
        if args.command == "status":
            code = asyncio.run(_run_status(config_path, getattr(args, "format", "text")))
            raise SystemExit(code)
        if args.command == "models":
            raise SystemExit(_run_models(args))
        asyncio.run(_run_orchestrator(config_path))
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Shutting down on keyboard interrupt.")


if __name__ == "__main__":
    main()
