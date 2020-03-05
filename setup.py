"""Install Mesh TensorFlow."""

from setuptools import find_packages
from setuptools import setup

setup(
    name='mesh-tensorflow',
    version='0.1.13',
    description='Mesh TensorFlow',
    author='Google Inc.',
    author_email='no-reply@google.com',
    url='http://github.com/tensorflow/mesh',
    license='Apache 2.0',
    packages=find_packages(),
    package_data={
        # Include gin files.
        'transformer': ['transformer/gin/*.gin'],
    },
    scripts=[],
    install_requires=[
        'absl-py',
        'future',
        'gin-config',
        'six',
    ],
    extras_require={
        'auto_mtf': ['ortools'],
        'tensorflow': ['tensorflow>=1.15.0'],
        'transformer': ['tensorflow-datasets'],
    },
    tests_require=[
        'ortools',
        'pytest',
        'tensorflow',
        'tensorflow-datasets',
    ],
    setup_requires=['pytest-runner'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='tensorflow machine learning',
)
