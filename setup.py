from setuptools import setup, find_packages

VERSION = "0.8"

REQUIRES = ["requests"]

setup(
    name="deepstack-python",
    version=VERSION,
    url="https://github.com/robmarkcole/deepstack-python",
    author="Robin Cole",
    author_email="robmarkcole@gmail.com",
    description="Unofficial python API for DeepStack",
    install_requires=REQUIRES,
    packages=find_packages(),
    license="Apache License, Version 2.0",
    python_requires=">=3.7",
    classifiers=[
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
)
