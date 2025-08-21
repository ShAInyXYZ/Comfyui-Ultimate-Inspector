# ComfyUI Ultimate Inspector

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A diagnostic tool for **ComfyUI** that scans your installation, Python environment, custom nodes, dependencies, and models.  
It generates a full markdown report to help with reproducibility, debugging, and sharing setups.

## Features

- 🔍 System and GPU information
- 🐍 Python environment and package list
- 📦 Custom node analysis with dependency checks
- 🧩 Model inventory and configuration paths
- 💡 Recommendations for missing or conflicting dependencies

## New Extended Features (COMING SOON)

- User interface for easier navigation
- More advanced diagnostics and fixes
- Extra reporting options and automation

## Structure

```
ComfyUI_portable/                 # Root portable folder
├── ComfyUI/                      # Main ComfyUI application
├── python_embeded/               # Python Runtime
└── ComfyUI-Ultimate-Inspector/   # Our repo
    └── ComfyUI-Ultimate-Inspector.py
```

## Installation

Clone the repo to your **ComfyUI_portable** :

```bash
git clone https://github.com/ShAlnyXYZ/ComfyUI-Ultimate-Inspector.git
```
## Usage
Run the tool from the **ComfyUI_portable root directory** (one level above the `ComfyUI` folder).

Option 1 – Python directly
```
python ComfyUI-Ultimate-Inspector/ComfyUI-Ultimate-Inspector.py --output report.md --verbose
```
Option 2 – One-click batch file
Run **Run_ComfyUI_Inspector.bat**

This creates a timestamped subfolder under reports/ containing both the markdown report and the pip_freeze.txt requirements file.

## License

This project is licensed under the [MIT License](LICENSE).
Copyright © 2025 Mounir Belahbib (ShAlnyXYZ)
