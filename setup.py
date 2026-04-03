from setuptools import find_packages, setup

setup(
    name="titanic-ml",
    packages=find_packages(),
    version="0.1.0",
    description="End-to-end ML pipeline for Titanic survival prediction",
    author="Ton Nom",
    license="MIT",
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "flask",
    ],
)