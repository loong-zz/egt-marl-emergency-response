from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="egt-marl-disaster",
    version="1.0.0",
    author="Research Team",
    author_email="research@example.com",
    description="Evolutionary Game Theory - Multi-Agent Reinforcement Learning for Disaster Medical Resource Allocation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/egt-marl-disaster",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.981",
            "pre-commit>=2.20.0",
        ],
        "gpu": [
            "torch>=1.12.0+cu113",
            "torchvision>=0.13.0+cu113",
            "torchaudio>=0.12.0+cu113",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "egtmarl-train=experiments.train_egt_marl:main",
            "egtmarl-eval=experiments.evaluate_baselines:main",
            "egtmarl-sim=environments.disaster_sim:main",
        ],
    },
    include_package_data=True,
    package_data={
        "egt_marl": [
            "configs/*.yaml",
            "data/*.csv",
            "models/*.pt",
        ],
    },
)