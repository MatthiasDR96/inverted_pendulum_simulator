import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="inverted_pendulum_simulator",
    version="0.0.1",
    author="Matthias De Ryck",
    author_email="matthias.deryck@kuleuven.be",
    description="A package for the simulation of an inverted pendulum and balancing robot with "
                "different control techniques",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MatthiasDR96/inverted_pendulum_simulator.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
