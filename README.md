# 🤖 AI File Butler

*An intelligent file organization tool that brings order to digital chaos*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 Overview

AI File Butler is a sophisticated file organization tool that uses artificial intelligence to automatically categorize, rename, and organize your files. Whether you have a chaotic Downloads folder, an unorganized photo collection, or scattered documents, the File Butler brings intelligent order to your digital life.

### Key Features

- 🧠 **Multiple AI Strategies**: Rule-based, ML clustering, and LLM-powered organization
- 🔍 **Smart Duplicate Detection**: Find exact and near-duplicate files
- 🛡️ **Safety First**: Mandatory dry-run mode with user approval
- 📊 **Rich Feature Extraction**: Content analysis, metadata parsing, and pattern recognition
- 🎛️ **Modular Architecture**: Pluggable engines and extractors
- 🖥️ **Multiple Interfaces**: CLI, GUI, and programmatic API
- 🎭 **Demo Mode**: Built-in test data generation for showcasing

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-file-butler.git
cd ai-file-butler

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Run the Demo

The easiest way to see the File Butler in action:

```bash
# Run interactive demo
python demo/run_demo.py

# Or use the CLI demo
file-butler demo
```

### Organize Your Files

```bash
# Dry run (safe preview)
file-butler