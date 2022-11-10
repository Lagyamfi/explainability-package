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

# This call to setup() does all the work
setup(
    name="explainability_package",
    version="0.1.0",
    description="Library with tools for working on explainability",
    #long_description=long_description,
    #long_description_content_type="text/markdown",
    #url="https://explainability-package.readthedocs.io/",
    author="Lawrence Adu-Gyamfi",
    author_email="lawrence.adu-gyamfi@cea.fr",
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
    packages=find_packages(include=["explainability-package", "explainability-package.*"]),
    include_package_data=True,
    package_dir=HERE,
    install_requires=[
        "numpy",
        "pandas",
        #"scikit-learn"
        ],
        python_requires='>=3.6',
)



"""from setuptools import setup

setup()
"""