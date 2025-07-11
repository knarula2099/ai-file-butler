from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-file-butler",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="An intelligent file organization tool powered by AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-file-butler",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: System :: Filesystems",
        "Topic :: Utilities",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "ml": ["scikit-learn>=1.0.0", "pandas>=1.3.0", "numpy>=1.21.0"],
        "llm": ["openai>=1.0.0", "langchain>=0.1.0"],
        "gui": ["streamlit>=1.20.0"],
        "dev": ["pytest>=6.0.0", "pytest-cov>=2.12.0", "black>=21.0.0", "flake8>=3.9.0"],
        "docs": ["mkdocs>=1.4.0", "mkdocs-material>=8.0.0"],
    },
    entry_points={
        "console_scripts": [
            "file-butler=file_butler.interfaces.cli:cli",
            "file-butler-demo=demo.run_demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "file_butler": ["*.yaml", "*.json"],
        "demo": ["*.py"],
    },
)
