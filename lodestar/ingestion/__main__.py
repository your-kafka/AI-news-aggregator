"""Run ingestion from the command line: `python -m lodestar.ingestion`."""

from __future__ import annotations

import json
import sys

from lodestar.core.logging import configure_logging
from lodestar.ingestion.runner import ingest


def main() -> int:
    configure_logging()
    result = ingest(trigger="cli")
    print(json.dumps(result, indent=2, default=str))
    return 0 if not result.get("failed_sources") else 1


if __name__ == "__main__":
    sys.exit(main())
