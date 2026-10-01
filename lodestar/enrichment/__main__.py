"""Drain the content queue: `python -m lodestar.enrichment [limit]`."""

from __future__ import annotations

import json
import sys

from lodestar.core.logging import configure_logging
from lodestar.enrichment.content import enrich_content


def main() -> int:
    configure_logging()
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    stats = enrich_content(limit=limit)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
