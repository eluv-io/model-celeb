# model-celeb

Celebrity face tagging in two tagger containers that share a vectorstore:

1. **[model-celeb-vector](model-celeb-vector)** (`celeb-vector`) detects faces in the media and emits one InsightFace r100 embedding per face. The tagger stores these in a vectorstore index.
2. **[model-celeb-vector-tagger](model-celeb-vector-tagger)** (`celeb-vector-tagger`) reads a content's face vectors back from the index, matches them against a ground truth pool (IBC by default), and emits celebrity tags.

Faces are embedded once, so a new or updated pool only reruns step 2.

## Layout

```
src/                          shared python package
  config.py                   config.yml loader used by both containers
  vector/                     face detection + embedding (model-celeb-vector)
  tagger/                     vectorstore client, pool matching, ground truth fetching (model-celeb-vector-tagger)
model-celeb-vector/           run.py, config.yml, Containerfile, build.sh
model-celeb-vector-tagger/    run.py, config.yml, Containerfile, build.sh
tests/                        test_vector.py, test_tagger.py
setup.py                      one package; install with the ".[vector]" or ".[tagger]" extra
Makefile.vector               make targets for celeb-vector
Makefile.tagger               make targets for celeb-vector-tagger
buildscripts/                 shared build tooling (git submodule)
```

## Build

Both containers build from the repo root:

```
git submodule update --init
make -f Makefile.vector build
make -f Makefile.tagger build
```

Each `build.sh` syncs the models its image needs into `models/`: the InsightFace weights for vector and the ground truth pools for tagger. Push with `make -f Makefile.<vector|tagger> deploy`.

Images are tagged `latest`. To build under another tag, pass it on the make command line, e.g. `make -f Makefile.vector build IMAGE_TAG=dev` (an `IMAGE_TAG` environment variable is overridden by buildscripts).

## Test

```
make -f Makefile.vector test     # runs the vector container over test-files/
make -f Makefile.tagger test     # tagger unit tests
pytest tests                     # both, in-process (vector needs the weights and a ".[vector]" env)
```

See each container's README for its output, runtime parameters and environment.
