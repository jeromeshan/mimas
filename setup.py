from setuptools import find_packages, setup
setup(
    name='mimaslib',
    packages=find_packages(include=['mimaslib']),
    version='0.1.0',
    description='',
    author='jeromeshan',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)