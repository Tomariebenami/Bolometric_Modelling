from os import path
import setuptools


here = path.abspath(path.dirname(__file__))

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open(path.join(here, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line for line in requirements_file.read().splitlines()
                    if not line.startswith('#')]


setuptools.setup(
    name="Bolometric_Modelling",
    version="0.0.0",
    author="Tom Ben-Ami",
    author_email="tomarie.benami@gmail.com",
    description="An analysis tool for modelling bolometric lightcurves of supernovae",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tomariebenami/Bolometric_Modelling",
    project_urls={
        "Bug Tracker": "https://github.com/Tomariebenami/Bolometric_Modelling/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU License",
        "Operating System :: OS Independent",
    ],
    #package_dir={"": "src"},
    packages=setuptools.find_packages(exclude=['docs', 'tests']),
    python_requires=">=3.6",
    install_requires=requirements,
)