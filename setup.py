from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tacit-tokenizer",
    version="2025.03.13",
    author="IansseiJelly",
    author_email="iansseijelly@berkeley.edu",
    description="A tokenizer for the Spike trace.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    project_urls={
        
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "",
    ],
    packages=find_packages(include=["tacit_learn"]),
    python_requires=">=3.10",
)