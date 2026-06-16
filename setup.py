from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="MLOPS-PROJECT-CODE",
    version="0.1",
    author="Aryan",
    packages=find_packages(),
    install_requires = requirements,
)