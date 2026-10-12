"""Run the supported `python -m sg_query_cache` command-line entry point."""

import sys

from sg_query_cache._cli import main

if __name__ == '__main__':
    sys.exit(main())
