import os
import sys

# make the repo's `src` package importable when running the tests from the source tree
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
