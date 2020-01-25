import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cost-sensitive-calibration",
    version="0.0.1",
    author="Tokukawa",
    author_email="emanuele.luzio@gmail.com",
    description="This package calibrate cost sensitive classifiers.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tokukawa/cost-sensitive-calibration",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
