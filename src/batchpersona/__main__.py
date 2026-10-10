"""CLI entrypoint allowing direct execution via `python -m batchpersona`."""

from __future__ import annotations

import sys

from batchpersona.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
