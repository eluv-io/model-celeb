from setuptools import setup

# Only needs model-celeb's `celeb.ground_truth` (pool fetching) from the parent repo: no face
# detection/embedding, so none of the mxnet/facenet/torch pins and no py3.9 ceiling.
setup(
    name='celeb_vector_tagger',
    version='0.1',
    packages=['celeb_vector_tagger', 'celeb'],
    package_dir={
        'celeb_vector_tagger': 'celeb_vector_tagger',
        'celeb': 'celeb',
    },
    python_requires='>=3.8',
    install_requires=[
        'numpy<2',
        'requests',
        'loguru',
        'setproctitle',
        'dacite',
        'PyYAML',
        'common-ml @ git+https://github.com/eluv-io/common-ml.git',
    ]
)
