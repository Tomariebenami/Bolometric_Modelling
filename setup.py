import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

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
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)