# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    
setup(
    name='pbgca',
    packages=find_packages(include=['pbgca']),
    version='0.1.0',
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
    description="Physically based galaxy clustering algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jeromeshan/pbgca",
    author="Andrey Shan",
    author_email="avshan@edu.hse.ru",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent"
    ],
    include_package_data=True,
    install_requires=["numpy","pandas","ray","scipy"]
)