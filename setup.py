from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

setup(
    name='Notq',
    version='1.0.0',
    description='Notq is a Python base tool collected and developed for speech and language processing in Persian',
    long_description=readme,
    author='Nbic',
    long_description_content_type="text/markdown",
    packages=find_packages(include=["Notq*"]),
    url="https://https://github.com/shaqayeql/Notq",
    install_requires=[],

    keywords=['python', 'first package'],
        classifiers= [
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
        ]
)