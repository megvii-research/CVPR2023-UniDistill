from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="Unidistill",
    version="0.0.1",
    author="Megvii",
    author_email="zhoushengchao@megvii.com",
    description="Model/Expriments for 3D Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=None,
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[],
)
