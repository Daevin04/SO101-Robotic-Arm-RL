"""Setup script for SO-101 Robotic Arm RL package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="so101-robotic-arm-rl",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Curriculum learning for robotic arm grasping with reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/SO101-Robotic-Arm-RL",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "so101-train=scripts.train:main",
            "so101-evaluate=scripts.evaluate:main",
            "so101-watch=scripts.watch:main",
        ],
    },
)
