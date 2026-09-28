from setuptools import setup

# One package for both containers; each installs only its extra:
#   pip install ".[vector]"   model-celeb-vector: detect + embed faces (mxnet/torch 1.9 pins, python 3.8-3.9)
#   pip install ".[tagger]"   model-celeb-vector-tagger: name stored face vectors against a pool (numpy only)
setup(
    name='celeb',
    version='0.2',
    packages=['src', 'src.vector', 'src.tagger'],
    python_requires='>=3.8',
    install_requires=[
        'dacite',
        'loguru',
        'PyYAML',
        'setproctitle',
        'common-ml @ git+https://github.com/eluv-io/common-ml.git',
    ],
    extras_require={
        'vector': [
            'opencv-python',
            'facenet_pytorch==2.5.2',
            'mxnet-cu101==1.9.1',
            'torch==1.9.0',
            'numpy<1.20.0',
        ],
        'tagger': [
            'numpy<2',
            'requests',
        ],
    },
)
