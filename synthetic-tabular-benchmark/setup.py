from setuptools import setup, find_packages

setup(
    name="synthetic-tabular-benchmark",
    version="0.1.0",
    description="Benchmark framework for synthetic tabular data models",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.26",
        "pandas>=2.2",
        "scikit-learn>=1.4",
        "pyyaml>=6.0.1",
        "click>=8.1",
        "tabulate>=0.9",
        "tqdm>=4.66",
        "openml>=0.14",
        "sdv>=1.12",
        "ctgan>=0.9",
        "faker>=26.0",
        "matplotlib>=3.8",
        "seaborn>=0.13",
    ],
    python_requires=">=3.9",
)


