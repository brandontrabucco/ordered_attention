"""Author: Brandon Trabucco, Copyright 2019"""


from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
    'tensorflow-gpu==2.5.3',
    'nltk',
    'matplotlib',
    'numpy'
]


setup(
    name='ordered_attention',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=[
        p for p in find_packages() if p.startswith('ordered_attention')
    ],
    description='An ordered multi-head attention mechanism.'
)
