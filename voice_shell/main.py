import argparse
import asyncio
import logging
from pathlib import Path

from voice_shell.src.config import Config
from voice_shell.src.orchestrator import Orchestrator


def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JarvOS Voice Shell PoC")
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
    return parser.parse_args()


async def _run(config_path: Path) -> None:
    config = Config.from_yaml(config_path) if config_path.exists() else Config()
    orchestrator = Orchestrator(config=config)
    await orchestrator.run()


def main() -> None:
    args = _parse_args()
    _configure_logging(args.log_level)

    try:
        asyncio.run(_run(args.config))
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Shutting down on keyboard interrupt.")


if __name__ == "__main__":
    main()
