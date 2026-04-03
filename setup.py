from setuptools import setup, find_packages

setup(
    name="toxic-classifier",
    version="0.1.0",
    description="Custom Transformer-based toxic comment classifier",
    author="Hoda Rajaei",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=0.24.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "toxic-train=scripts.train:main",
        ]
    },
)
