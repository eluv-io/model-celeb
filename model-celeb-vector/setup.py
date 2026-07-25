from setuptools import setup

# Same dependency set as model-celeb/setup.py: reuses model-celeb's `celeb` package (FaceModel's InsightFace path).
setup(
    name='celeb_vector',
    version='0.1',
    packages=['celeb_vector'],
    install_requires=[
        'opencv-python',
        'easydict==1.9',
        'facenet_pytorch==2.5.2',
        'mxnet-cu101==1.9.1',
        'networkx==2.6.3',
        'pandas==1.3.5',
        'scikit_learn==1.0.2',
        'scikit-image==0.17.2',
        'torch==1.9.0',
        'loguru',
        'setproctitle',
        'numpy<1.20.0',
        'dacite',
        'common-ml @ git+https://github.com/eluv-io/common-ml@vector-tags' # uses branch with vector tag support
    ]
)
