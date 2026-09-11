import os
import sys

# celeb_vector imports `celeb` (model-celeb's FaceModel) as an installed package. When running
# these tests against the source tree rather than an install, make the parent model-celeb repo
# importable so `celeb` (and `config`) resolve without installing anything.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # model-celeb-vector/
for path in (_REPO, os.path.dirname(_REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)
