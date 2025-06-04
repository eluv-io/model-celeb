from setuptools import setup

setup(
    name='my_package',
    version='0.1',
    packages=['celeb'],
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
        'numpy<1.20.0',
        'common_ml @ git+ssh://git@github.com/eluv-io/common-ml.git#egg=common_ml',
        'quick_test_py @ git+https://github.com/eluv-io/quick-test-py.git#egg=quick_test_py'
    ]
)