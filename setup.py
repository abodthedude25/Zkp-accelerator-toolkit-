"""Setup script for ZKP Accelerator Toolkit."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="zkp-accelerator-toolkit",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Educational toolkit for understanding ZKP hardware accelerators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/zkp-accelerator-toolkit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Security :: Cryptography",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "tabulate>=0.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "zkp-demo=src.main:main",
            "zkp-visualize=src.visualizer.demo:main",
            "zkp-simulate=src.simulator.demo:main",
            "zkp-optimize=src.optimizer.demo:main",
        ],
    },
)
