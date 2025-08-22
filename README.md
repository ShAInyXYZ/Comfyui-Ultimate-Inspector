# ComfyUI Ultimate Inspector

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A diagnostic tool for **ComfyUI** that scans your installation, Python environment, custom nodes, dependencies, and models.  
It generates a full markdown report to help with reproducibility, debugging, and sharing setups.

## Features

### 🔍 Core Analysis
- **System and GPU information** - Complete hardware and driver analysis
- **Python environment and package list** - Exact package versions with `pip list` method
- **Custom node analysis** with dependency checks and version conflicts
- **Model inventory** and configuration paths (default + extra_model_paths.yaml)
- **Recommendations** for missing or conflicting dependencies

### 🐳 **NEW: Docker Generator** 
- **Intelligent Docker container generation** from analysis reports
- **Smart image selection** with 60+ optimized base images (Python 3.9-3.13, CUDA 11.8-13.0, PyTorch 2.0-2.8)
- **Compatibility scoring system** that automatically picks the best Docker image for your setup
- **Complete deployment package** with Dockerfile, startup script, docker-compose.yml, and comprehensive documentation
- **Framework-aware optimization** - avoids reinstalling pre-built PyTorch/TensorFlow
- **Automatic model/custom_nodes copying** from your existing ComfyUI installation

### 🚀 Extended Features
- **Timestamped reports** with requirements_{day-hourminute}.txt format
- **Enhanced analysis** with ComfyUI-Manager integration
- **Model size tracking** and storage optimization recommendations

## Project Structure

```
ComfyUI_portable/                      # Root portable folder
├── ComfyUI/                           # Main ComfyUI application
├── python_embeded/                    # Python Runtime
└── ComfyUI-Ultimate-Inspector/        # Our repo
    ├── ComfyUI-Ultimate-Inspector.py  # Main analysis tool
    ├── Run_ComfyUI_Inspector.bat      # One-click analysis
    ├── reports/                       # Generated analysis reports
    │   └── XX-XXXX/                   # Timestamped report folders
    │       ├── Report_Comfy_XX-XXXX.md      # Comprehensive analysis
    │       └── requirements_XX-XXXX.txt     # Exact package list
    └── Comfyui-Docker-Generator/      # 🐳 NEW: Docker container generator
        ├── ComfyUI-Docker-Generator.py       # Intelligent Docker generator
        ├── docker_images_db.json             # 60+ optimized base images
        ├── Run_ComfyUI_Docker_Generator.bat  # One-click Docker generation
        └── README.md                         # Docker generator documentation
```

## Installation

Clone the repo to your **ComfyUI_portable** :

```bash
git clone https://github.com/ShAlnyXYZ/ComfyUI-Ultimate-Inspector.git
```
## Usage

### 🔍 Step 1: Analyze Your ComfyUI Environment

Run the analysis tool from the **ComfyUI_portable root directory** (one level above the `ComfyUI` folder).

**Option A – One-click batch file (Recommended)**
```bash
# Double-click or run in terminal
Run_ComfyUI_Inspector.bat
```

**Option B – Python directly**
```bash
python ComfyUI-Ultimate-Inspector/ComfyUI-Ultimate-Inspector.py --output report.md --verbose
```

This creates a timestamped subfolder under `reports/XX-XXXX/` containing:
- `Report_Comfy_XX-XXXX.md` - Comprehensive analysis report
- `requirements_XX-XXXX.txt` - Exact package list using improved `pip list` method

### 🐳 Step 2: Generate Docker Container (NEW!)

After generating your analysis report, create a portable Docker container:

**Option A – One-click batch file (Recommended)**
```bash
# Navigate to Docker generator
cd ComfyUI-Ultimate-Inspector/Comfyui-Docker-Generator

# Double-click or run the batch file
Run_ComfyUI_Docker_Generator.bat
```

**Option B – Python directly**
```bash
# Navigate to Docker generator  
cd ComfyUI-Ultimate-Inspector/Comfyui-Docker-Generator

# Run the intelligent Docker generator
python ComfyUI-Docker-Generator.py

# Follow the interactive prompts:
# 1. Select your report directory (auto-detected)
# 2. Review environment analysis
# 3. Choose optimal Docker base image (scored recommendations)
# 4. Generate complete Docker package
```

### 🚀 Step 3: Deploy Your Container

```bash
# Navigate to generated Docker folder
cd docker_build_XX-XXXX

# Deploy with automated script (copies models + custom_nodes)
bash startup.sh

# OR use Docker Compose
docker-compose up -d

# Access ComfyUI at: http://localhost:8188
```

---

## 🎯 Complete Workflow

```mermaid
graph LR
    A[ComfyUI Environment] --> B[Run Inspector]
    B --> C[Generate Report]
    C --> D[Run Docker Generator]
    D --> E[Select Base Image]
    E --> F[Generate Container]
    F --> G[Deploy Anywhere]
```

1. **Analyze** → Run Ultimate Inspector to scan your ComfyUI setup
2. **Containerize** → Use Docker Generator to create optimized containers
3. **Deploy** → Run your ComfyUI environment anywhere with Docker

---

## 🌟 What's New in This Version

### Enhanced Analysis Engine
- **Fixed pip freeze issues** - Now uses `pip list` for accurate package detection
- **Improved CUDA detection** - Better GPU and PyTorch version analysis
- **ComfyUI-Manager integration** - Analyzes Manager databases and snapshots

### 🐳 Docker Generator (Major New Feature)
- **Intelligent image selection** from 60+ optimized base images
- **Compatibility scoring** automatically picks the best Docker image
- **2024-2025 versions** - CUDA 13.0, PyTorch 2.8.0, Python 3.13 support
- **Framework optimization** - Avoids conflicts with pre-installed libraries
- **Complete deployment package** - Everything needed to run anywhere

### Improved Output
- **Timestamped filenames** - `requirements_day-hourminute.txt` format
- **Better organization** - Separate folders for analysis and Docker tools
- **Enhanced documentation** - Comprehensive guides for both tools

## License

This project is licensed under the [MIT License](LICENSE).
Copyright © 2025 Mounir Belahbib (ShAlnyXYZ)
