"""
ComfyUI Docker Container Generator
Generates Docker containers from ComfyUI Ultimate Inspector reports
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from datetime import datetime
import shutil

class ComfyUIDockerGenerator:
    def __init__(self, database_path="docker_images_db.json"):
        self.database_path = database_path
        self.database = self.load_database()
        self.report_data = {}
        self.docker_config = {}
        
    def load_database(self):
        """Load the Docker images database"""
        try:
            with open(self.database_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[!] Error loading database: {e}")
            return {}
    
    def find_report_directories(self, base_path="."):
        """Find report directories with pattern /reports/xx-xxx/"""
        reports_dir = Path(base_path) / "reports"
        if not reports_dir.exists():
            return []
        
        report_dirs = []
        for item in reports_dir.iterdir():
            if item.is_dir():
                # Look for pattern like "22-1435" or similar timestamp formats
                if re.match(r'\d{2}-\d{3,4}', item.name):
                    # Check if it contains .md and .txt files
                    md_files = list(item.glob("*.md"))
                    txt_files = list(item.glob("requirements_*.txt"))
                    if md_files and txt_files:
                        report_dirs.append({
                            'path': item,
                            'name': item.name,
                            'md_file': md_files[0],
                            'txt_file': txt_files[0],
                            'timestamp': item.name
                        })
        
        return sorted(report_dirs, key=lambda x: x['timestamp'], reverse=True)
    
    def select_report_directory(self, base_path="."):
        """Interactive selection of report directory"""
        print(f"[*] Scanning for report directories in: {Path(base_path).resolve()}")
        
        report_dirs = self.find_report_directories(base_path)
        
        if not report_dirs:
            print("[!] No report directories found.")
            print("[!] Expected structure: /reports/XX-XXXX/ containing .md and requirements_*.txt files")
            return None
        
        print(f"\n[+] Found {len(report_dirs)} report directories:")
        for i, report in enumerate(report_dirs, 1):
            print(f"  {i}. {report['name']} - {report['md_file'].name} + {report['txt_file'].name}")
        
        while True:
            try:
                choice = input(f"\nSelect report directory (1-{len(report_dirs)}) [1]: ").strip()
                if not choice:
                    choice = "1"
                idx = int(choice) - 1
                if 0 <= idx < len(report_dirs):
                    selected = report_dirs[idx]
                    print(f"[+] Selected: {selected['name']}")
                    return selected
                else:
                    print(f"[!] Please enter a number between 1 and {len(report_dirs)}")
            except ValueError:
                print("[!] Please enter a valid number")
            except KeyboardInterrupt:
                print("\n[!] Cancelled by user")
                return None
    
    def parse_markdown_report(self, md_file):
        """Parse the markdown report to extract system information"""
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            data = {
                'system': {},
                'python': {},
                'comfyui': {},
                'gpu_info': {},
                'packages': [],
                'models_info': {},
                'custom_nodes': []
            }
            
            # Extract system information
            system_section = self.extract_section(content, "## 🔧 System Information")
            if system_section:
                data['system'] = self.parse_system_info(system_section)
            
            # Extract Python environment
            python_section = self.extract_section(content, "## 🐍 Python Environment")
            if python_section:
                data['python'] = self.parse_python_info(python_section)
            
            # Extract ComfyUI installation
            comfyui_section = self.extract_section(content, "## 🎨 ComfyUI Installation")
            if comfyui_section:
                data['comfyui'] = self.parse_comfyui_info(comfyui_section)
            
            # Extract GPU information from system section
            data['gpu_info'] = self.parse_gpu_info(system_section if system_section else "")
            
            # Extract model inventory
            models_section = self.extract_section(content, "### 🎭 Model Inventory")
            if models_section:
                data['models_info'] = self.parse_models_info(models_section)
            
            return data
            
        except Exception as e:
            print(f"[!] Error parsing markdown report: {e}")
            return {}
    
    def extract_section(self, content, section_header):
        """Extract a specific section from markdown content"""
        lines = content.split('\n')
        section_lines = []
        in_section = False
        
        for line in lines:
            if line.strip().startswith(section_header):
                in_section = True
                continue
            elif in_section and line.strip().startswith('##') and not line.strip().startswith('###'):
                break
            elif in_section:
                section_lines.append(line)
        
        return '\n'.join(section_lines) if section_lines else None
    
    def parse_system_info(self, section):
        """Parse system information section"""
        info = {}
        for line in section.split('\n'):
            if '**Platform:**' in line:
                info['platform'] = line.split('**Platform:**')[1].strip()
            elif '**Architecture:**' in line:
                info['architecture'] = line.split('**Architecture:**')[1].strip()
            elif '**RAM:**' in line:
                info['ram'] = line.split('**RAM:**')[1].strip()
        return info
    
    def parse_python_info(self, section):
        """Parse Python environment section"""
        info = {}
        for line in section.split('\n'):
            line = line.strip()
            if '**Path:**' in line:
                info['path'] = line.split('**Path:**')[1].strip().replace('`', '')
            elif '**Version:**' in line and 'version' not in info:  # Only take first version
                info['version'] = line.split('**Version:**')[1].strip()
            elif '**Environment Type:**' in line:
                info['env_type'] = line.split('**Environment Type:**')[1].strip()
        return info
    
    def parse_comfyui_info(self, section):
        """Parse ComfyUI installation section"""
        info = {}
        for line in section.split('\n'):
            if '**Path:**' in line:
                info['path'] = line.split('**Path:**')[1].strip().replace('`', '')
            elif '**Version:**' in line:
                info['version'] = line.split('**Version:**')[1].strip()
            elif '**Branch:**' in line:
                info['branch'] = line.split('**Branch:**')[1].strip()
            elif '**Remote:**' in line:
                info['remote'] = line.split('**Remote:**')[1].strip()
        return info
    
    def parse_gpu_info(self, section):
        """Parse GPU information from system section"""
        info = {'has_nvidia': False, 'has_cuda': False}
        
        if 'NVIDIA' in section:
            info['has_nvidia'] = True
        if 'CUDA' in section or 'PyTorch CUDA' in section:
            info['has_cuda'] = True
            
        return info
    
    def parse_models_info(self, section):
        """Parse model inventory section"""
        info = {'total_models': 0, 'total_size_gb': 0.0}
        
        for line in section.split('\n'):
            if '**Total Models:**' in line:
                # Extract number and size from "123 files (45.6 GB)"
                match = re.search(r'(\d+)\s+files\s+\(([0-9.]+)\s+GB\)', line)
                if match:
                    info['total_models'] = int(match.group(1))
                    info['total_size_gb'] = float(match.group(2))
        
        return info
    
    def load_requirements_file(self, txt_file):
        """Load and parse requirements.txt file"""
        packages = []
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        packages.append(line)
            return packages
        except Exception as e:
            print(f"[!] Error loading requirements file: {e}")
            return []
    
    def analyze_report_requirements(self, report_data, packages):
        """Analyze report to determine optimal Docker configuration"""
        analysis = {
            'python_version': None,
            'pytorch_detected': False,
            'tensorflow_detected': False,
            'cuda_version': None,
            'gpu_required': False,
            'has_nvidia': False,
            'recommended_category': 'cpu_only',
            'package_analysis': {}
        }
        
        # Analyze Python version from report
        python_info = report_data.get('python', {})
        
        if 'version' in python_info:
            version_string = python_info['version']
            
            # Try multiple patterns to match Python version
            version_match = re.search(r'Python (\d+\.\d+)', version_string)
            if not version_match:
                # Try pattern without "Python" prefix - match X.Y.Z and extract X.Y
                version_match = re.search(r'(\d+\.\d+)\.?\d*', version_string)
            
            if version_match:
                analysis['python_version'] = version_match.group(1)
        
        # Analyze GPU information
        gpu_info = report_data.get('gpu_info', {})
        analysis['has_nvidia'] = gpu_info.get('has_nvidia', False)
        analysis['gpu_required'] = gpu_info.get('has_cuda', False)
        
        # Analyze packages for ML frameworks
        pytorch_packages = ['torch', 'pytorch', 'torchvision', 'torchaudio']
        tensorflow_packages = ['tensorflow', 'tensorflow-gpu', 'tf-nightly']
        
        for package in packages:
            package_lower = package.lower()
            if any(p in package_lower for p in pytorch_packages):
                analysis['pytorch_detected'] = True
                # Extract CUDA version from package if available
                cuda_match = re.search(r'cu(\d+)', package_lower)
                if cuda_match:
                    cuda_version = cuda_match.group(1)
                    if cuda_version == '121':
                        analysis['cuda_version'] = '12.1'
                    elif cuda_version == '118':
                        analysis['cuda_version'] = '11.8'
                    elif cuda_match.group(1).startswith('12'):
                        analysis['cuda_version'] = '12.x'
            
            if any(p in package_lower for p in tensorflow_packages):
                analysis['tensorflow_detected'] = True
        
        # Determine recommended category
        if analysis['gpu_required']:
            if analysis['pytorch_detected']:
                analysis['recommended_category'] = 'pytorch_optimized'
            elif analysis['tensorflow_detected']:
                analysis['recommended_category'] = 'tensorflow_optimized'
            else:
                analysis['recommended_category'] = 'gpu_required'
        else:
            analysis['recommended_category'] = 'cpu_only'
        
        return analysis
    
    def get_compatible_images(self, analysis):
        """Get compatible Docker images based on analysis"""
        compatible_images = []
        
        # Get image categories from database
        image_categories = self.database.get('docker_base_images', {})
        compatibility_matrix = self.database.get('compatibility_matrix', {})
        
        for category_name, category_data in image_categories.items():
            for image_name, image_info in category_data.get('images', {}).items():
                is_compatible = True
                compatibility_score = 0
                reasons = []
                
                # Check Python version compatibility
                if analysis['python_version']:
                    image_python = image_info.get('python_version')
                    if image_python:
                        if image_python == analysis['python_version']:
                            compatibility_score += 10
                            reasons.append(f"Python {image_python} match")
                        elif abs(float(image_python) - float(analysis['python_version'])) <= 0.1:
                            compatibility_score += 5
                            reasons.append(f"Python {image_python} compatible")
                        else:
                            compatibility_score -= 5
                            reasons.append(f"Python {image_python} version difference")
                
                # Check GPU requirements
                if analysis['gpu_required']:
                    if image_info.get('gpu_support', False):
                        compatibility_score += 15
                        reasons.append("GPU support available")
                    else:
                        is_compatible = False
                        reasons.append("No GPU support")
                else:
                    if not image_info.get('gpu_support', False):
                        compatibility_score += 5
                        reasons.append("CPU-only optimized")
                
                # Check framework compatibility
                if analysis['pytorch_detected']:
                    pytorch_version = image_info.get('pytorch_version')
                    if pytorch_version:
                        compatibility_score += 20
                        reasons.append(f"PyTorch {pytorch_version} included")
                    elif category_name in ['nvidia_cuda', 'python_official']:
                        compatibility_score += 5
                        reasons.append("Can install PyTorch")
                
                if analysis['tensorflow_detected']:
                    tf_version = image_info.get('tensorflow_version')
                    if tf_version:
                        compatibility_score += 20
                        reasons.append(f"TensorFlow {tf_version} included")
                
                # Check CUDA compatibility
                if analysis['cuda_version']:
                    image_cuda = image_info.get('cuda_version')
                    if image_cuda:
                        if analysis['cuda_version'] in str(image_cuda):
                            compatibility_score += 15
                            reasons.append(f"CUDA {image_cuda} match")
                        else:
                            compatibility_score += 5
                            reasons.append(f"CUDA {image_cuda} available")
                
                # Boost score for recommended images
                if image_info.get('recommended'):
                    compatibility_score += 10
                    reasons.append("Officially recommended")
                
                # Boost score for latest/stable status
                status = image_info.get('status', 'unknown')
                if status == 'latest':
                    compatibility_score += 8
                elif status == 'stable':
                    compatibility_score += 5
                
                if is_compatible:
                    compatible_images.append({
                        'name': image_name,
                        'info': image_info,
                        'category': category_name,
                        'score': compatibility_score,
                        'reasons': reasons
                    })
        
        # Sort by compatibility score (highest first)
        compatible_images.sort(key=lambda x: x['score'], reverse=True)
        return compatible_images
    
    def select_docker_base_image(self, report_data, packages):
        """Intelligent selection of Docker base image based on report analysis"""
        print("\n[*] Analyzing report for optimal Docker image selection...")
        
        # Analyze requirements from report
        analysis = self.analyze_report_requirements(report_data, packages)
        
        print(f"[+] Analysis results:")
        print(f"    Python version: {analysis['python_version'] or 'Unknown'}")
        print(f"    GPU required: {'Yes' if analysis['gpu_required'] else 'No'}")
        print(f"    PyTorch detected: {'Yes' if analysis['pytorch_detected'] else 'No'}")
        print(f"    TensorFlow detected: {'Yes' if analysis['tensorflow_detected'] else 'No'}")
        if analysis['cuda_version']:
            print(f"    CUDA version: {analysis['cuda_version']}")
        
        # Get compatible images
        compatible_images = self.get_compatible_images(analysis)
        
        if not compatible_images:
            print("[!] No compatible images found in database")
            return None, None
        
        # Show top recommendations
        print(f"\n[+] Found {len(compatible_images)} compatible images")
        print("Top recommendations:")
        
        # Group by categories for better display
        displayed_images = []
        categories_shown = set()
        
        # First, show the top 5 overall
        top_images = compatible_images[:5]
        
        # Then add one from each major category if not already shown
        for img in compatible_images:
            if img['category'] not in categories_shown and len(displayed_images) < 8:
                if img not in top_images:
                    displayed_images.append(img)
                categories_shown.add(img['category'])
        
        # Combine top images with category representatives
        final_images = top_images + displayed_images
        
        # Remove duplicates while preserving order
        seen = set()
        unique_images = []
        for img in final_images:
            if img['name'] not in seen:
                unique_images.append(img)
                seen.add(img['name'])
        
        # Limit to top 6 for manageable selection
        display_images = unique_images[:6]
        
        for i, img in enumerate(display_images, 1):
            score_bar = "★" * min(5, max(1, img['score'] // 5))
            rec_mark = "⭐" if img['info'].get('recommended') else ""
            
            print(f"  {i}. {img['name']} {rec_mark}")
            print(f"     {img['info']['description']}")
            print(f"     Size: {img['info']['size']} | Score: {score_bar} ({img['score']})")
            print(f"     Reasons: {', '.join(img['reasons'][:3])}")
            print()
        
        # Interactive selection
        while True:
            try:
                choice = input(f"Select base image (1-{len(display_images)}) [1]: ").strip()
                if not choice:
                    choice = "1"
                idx = int(choice) - 1
                if 0 <= idx < len(display_images):
                    selected = display_images[idx]
                    print(f"[+] Selected: {selected['name']} (Score: {selected['score']})")
                    print(f"[+] Selection reasons: {', '.join(selected['reasons'])}")
                    return selected['name'], selected['info']
                else:
                    print(f"[!] Please enter a number between 1 and {len(display_images)}")
            except ValueError:
                print("[!] Please enter a valid number")
            except KeyboardInterrupt:
                print("\n[!] Cancelled by user")
                return None, None
    
    def generate_dockerfile(self, base_image, image_info, packages, comfyui_info, output_dir):
        """Generate optimized Dockerfile based on report data and image info"""
        dockerfile_content = []
        
        # Base image with metadata
        dockerfile_content.append(f"# Generated Dockerfile for ComfyUI")
        dockerfile_content.append(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        dockerfile_content.append(f"# Base image: {base_image}")
        dockerfile_content.append(f"# Image info: {image_info.get('description', 'Custom image')}")
        if image_info.get('cuda_version'):
            dockerfile_content.append(f"# CUDA version: {image_info['cuda_version']}")
        if image_info.get('pytorch_version'):
            dockerfile_content.append(f"# PyTorch version: {image_info['pytorch_version']}")
        dockerfile_content.append("")
        dockerfile_content.append(f"FROM {base_image}")
        dockerfile_content.append("")
        
        # Environment variables (enhanced based on image type)
        dockerfile_content.append("# Environment setup")
        dockerfile_content.append("ENV DEBIAN_FRONTEND=noninteractive")
        dockerfile_content.append("ENV PYTHONUNBUFFERED=1")
        dockerfile_content.append("ENV PYTHONDONTWRITEBYTECODE=1")
        
        # Add CUDA-specific environment variables if GPU image
        if image_info.get('gpu_support', False):
            dockerfile_content.append("ENV NVIDIA_VISIBLE_DEVICES=all")
            dockerfile_content.append("ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility")
        
        dockerfile_content.append("")
        
        # System packages (optimized based on base image)
        dockerfile_content.append("# Install system dependencies")
        
        # Check if we need to install system packages (some images have them pre-installed)
        needs_system_packages = True
        if 'pytorch/pytorch' in base_image or 'tensorflow' in base_image or 'nvcr.io' in base_image:
            needs_system_packages = False
        
        if needs_system_packages and ('ubuntu' in base_image or 'debian' in base_image):
            dockerfile_content.append("RUN apt-get update && apt-get install -y \\")
            dockerfile_content.append("    curl \\")
            dockerfile_content.append("    wget \\") 
            dockerfile_content.append("    git \\")
            dockerfile_content.append("    build-essential \\")
            
            # Add Python packages only if not already in image
            if 'python:' not in base_image:
                dockerfile_content.append("    python3 \\")
                dockerfile_content.append("    python3-dev \\")
                dockerfile_content.append("    python3-pip \\")
            
            dockerfile_content.append("    libgl1-mesa-glx \\")
            dockerfile_content.append("    libglib2.0-0 \\")
            dockerfile_content.append("    libsm6 \\")
            dockerfile_content.append("    libxext6 \\")
            dockerfile_content.append("    libxrender-dev \\")
            dockerfile_content.append("    libgomp1 \\")
            dockerfile_content.append("    && rm -rf /var/lib/apt/lists/*")
        elif needs_system_packages:
            dockerfile_content.append("# Note: System packages may need to be installed manually for this base image")
        
        dockerfile_content.append("")
        
        # Working directory
        dockerfile_content.append("# Set working directory")
        dockerfile_content.append("WORKDIR /app")
        dockerfile_content.append("")
        
        # Python pip upgrade (conditional)
        has_preinstalled_frameworks = any(key in image_info for key in ['pytorch_version', 'tensorflow_version'])
        if not has_preinstalled_frameworks:
            dockerfile_content.append("# Upgrade pip and install Python requirements")
            dockerfile_content.append("COPY requirements.txt .")
            dockerfile_content.append("RUN pip install --no-cache-dir --upgrade pip")
            dockerfile_content.append("RUN pip install --no-cache-dir -r requirements.txt")
        else:
            dockerfile_content.append("# Install additional Python requirements (frameworks pre-installed)")
            dockerfile_content.append("COPY requirements.txt .")
            dockerfile_content.append("# Filter out pre-installed packages to avoid conflicts")
            dockerfile_content.append("RUN grep -v -i 'torch\\|tensorflow' requirements.txt > requirements_filtered.txt || echo '# No additional packages' > requirements_filtered.txt")
            dockerfile_content.append("RUN pip install --no-cache-dir --upgrade pip")
            dockerfile_content.append("RUN if [ -s requirements_filtered.txt ]; then pip install --no-cache-dir -r requirements_filtered.txt; fi")
        
        dockerfile_content.append("")
        
        # Clone ComfyUI
        comfyui_remote = comfyui_info.get('remote', 'https://github.com/comfyanonymous/ComfyUI.git')
        comfyui_branch = comfyui_info.get('branch', 'master')
        dockerfile_content.append("# Clone ComfyUI")
        dockerfile_content.append(f"RUN git clone {comfyui_remote} ComfyUI")
        dockerfile_content.append(f"RUN cd ComfyUI && git checkout {comfyui_branch}")
        dockerfile_content.append("")
        
        # Create necessary directories
        dockerfile_content.append("# Create directories for models and custom nodes")
        dockerfile_content.append("RUN mkdir -p /app/ComfyUI/models")
        dockerfile_content.append("RUN mkdir -p /app/ComfyUI/custom_nodes")
        dockerfile_content.append("RUN mkdir -p /app/ComfyUI/input")
        dockerfile_content.append("RUN mkdir -p /app/ComfyUI/output")
        dockerfile_content.append("")
        
        # Expose port
        dockerfile_content.append("# Expose ComfyUI port")
        dockerfile_content.append("EXPOSE 8188")
        dockerfile_content.append("")
        
        # Default command
        dockerfile_content.append("# Default command")
        dockerfile_content.append("CMD [\"python\", \"ComfyUI/main.py\", \"--listen\", \"--port\", \"8188\"]")
        
        # Write Dockerfile
        dockerfile_path = Path(output_dir) / "Dockerfile"
        with open(dockerfile_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(dockerfile_content))
        
        print(f"[+] Optimized Dockerfile generated: {dockerfile_path}")
        print(f"[+] Optimizations: {'GPU support' if image_info.get('gpu_support') else 'CPU-only'}, " +
              f"{'Pre-built frameworks' if has_preinstalled_frameworks else 'Custom install'}")
        return dockerfile_path
    
    def generate_startup_script(self, comfyui_path, models_info, output_dir):
        """Generate startup.sh script for copying models and custom nodes"""
        script_content = []
        
        script_content.append("#!/bin/bash")
        script_content.append(f"# ComfyUI Docker Startup Script")
        script_content.append(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        script_content.append("")
        script_content.append("set -e")
        script_content.append("")
        
        script_content.append("echo '[*] ComfyUI Docker Container Setup'")
        script_content.append("")
        
        # Container name
        container_name = f"comfyui-container-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        script_content.append(f"CONTAINER_NAME=\"{container_name}\"")
        script_content.append("IMAGE_NAME=\"comfyui-generated\"")
        script_content.append("")
        
        # Build Docker image
        script_content.append("echo '[*] Building Docker image...'")
        script_content.append("docker build -t $IMAGE_NAME .")
        script_content.append("")
        
        # Run container
        script_content.append("echo '[*] Starting Docker container...'")
        script_content.append("docker run -d --name $CONTAINER_NAME \\")
        script_content.append("  -p 8188:8188 \\")
        
        # Add GPU support if needed
        if models_info.get('gpu_required', False):
            script_content.append("  --gpus all \\")
        
        script_content.append("  $IMAGE_NAME")
        script_content.append("")
        
        script_content.append("echo '[*] Waiting for container to start...'")
        script_content.append("sleep 5")
        script_content.append("")
        
        # Copy models directory
        if comfyui_path and Path(comfyui_path).exists():
            models_path = Path(comfyui_path) / "models"
            custom_nodes_path = Path(comfyui_path) / "custom_nodes"
            
            if models_path.exists():
                script_content.append("echo '[*] Copying models directory...'")
                script_content.append(f"docker cp \"{models_path}\" $CONTAINER_NAME:/app/ComfyUI/")
                script_content.append("")
            
            if custom_nodes_path.exists():
                script_content.append("echo '[*] Copying custom_nodes directory...'")
                script_content.append(f"docker cp \"{custom_nodes_path}\" $CONTAINER_NAME:/app/ComfyUI/")
                script_content.append("")
        
        # Restart container to apply changes
        script_content.append("echo '[*] Restarting container to apply changes...'")
        script_content.append("docker restart $CONTAINER_NAME")
        script_content.append("")
        
        script_content.append("echo '[+] Setup complete!'")
        script_content.append("echo '[+] ComfyUI should be available at: http://localhost:8188'")
        script_content.append("echo '[+] Container name: '$CONTAINER_NAME")
        script_content.append("")
        script_content.append("echo 'Useful commands:'")
        script_content.append("echo '  View logs: docker logs '$CONTAINER_NAME")
        script_content.append("echo '  Stop container: docker stop '$CONTAINER_NAME")
        script_content.append("echo '  Remove container: docker rm '$CONTAINER_NAME")
        
        # Write startup script
        script_path = Path(output_dir) / "startup.sh"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(script_content))
        
        # Make executable on Unix systems
        if os.name != 'nt':  # Not Windows
            script_path.chmod(0o755)
        
        print(f"[+] Startup script generated: {script_path}")
        return script_path
    
    def generate_docker_compose(self, models_info, output_dir):
        """Generate docker-compose.yml for easier management"""
        compose_content = {
            "version": "3.8",
            "services": {
                "comfyui": {
                    "build": ".",
                    "ports": ["8188:8188"],
                    "volumes": [
                        "./models:/app/ComfyUI/models",
                        "./custom_nodes:/app/ComfyUI/custom_nodes", 
                        "./input:/app/ComfyUI/input",
                        "./output:/app/ComfyUI/output"
                    ],
                    "environment": [
                        "PYTHONUNBUFFERED=1"
                    ],
                    "restart": "unless-stopped"
                }
            }
        }
        
        # Add GPU support if needed
        if models_info.get('gpu_required', False):
            compose_content["services"]["comfyui"]["deploy"] = {
                "resources": {
                    "reservations": {
                        "devices": [{
                            "driver": "nvidia",
                            "count": "all",
                            "capabilities": ["gpu"]
                        }]
                    }
                }
            }
        
        import yaml
        compose_path = Path(output_dir) / "docker-compose.yml"
        with open(compose_path, 'w', encoding='utf-8') as f:
            yaml.dump(compose_content, f, default_flow_style=False, sort_keys=False)
        
        print(f"[+] Docker Compose file generated: {compose_path}")
        return compose_path
    
    def generate_readme(self, base_image, image_info, packages, comfyui_info, models_info, report_data, analysis, output_dir):
        """Generate comprehensive README.md with setup instructions and information"""
        readme_content = []
        
        # Header
        readme_content.append("# 🐳 ComfyUI Docker Container")
        readme_content.append("")
        readme_content.append("**Auto-generated Docker setup for ComfyUI**")
        readme_content.append("")
        readme_content.append(f"Generated on: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
        readme_content.append("")
        readme_content.append("---")
        readme_content.append("")
        
        # Quick Start
        readme_content.append("## 🚀 Quick Start")
        readme_content.append("")
        readme_content.append("```bash")
        readme_content.append("# Build and run with the automated script")
        readme_content.append("bash startup.sh")
        readme_content.append("")
        readme_content.append("# OR use Docker Compose")
        readme_content.append("docker-compose up -d")
        readme_content.append("")
        readme_content.append("# Access ComfyUI at:")
        readme_content.append("# http://localhost:8188")
        readme_content.append("```")
        readme_content.append("")
        
        # Configuration Summary
        readme_content.append("## ⚙️ Configuration Summary")
        readme_content.append("")
        readme_content.append("| Component | Value |")
        readme_content.append("|-----------|-------|")
        readme_content.append(f"| **Base Image** | `{base_image}` |")
        readme_content.append(f"| **Image Description** | {image_info.get('description', 'Custom Docker image')} |")
        readme_content.append(f"| **Image Size** | {image_info.get('size', 'Unknown')} |")
        readme_content.append(f"| **GPU Support** | {'✅ Yes' if image_info.get('gpu_support') else '❌ No'} |")
        
        if image_info.get('cuda_version'):
            readme_content.append(f"| **CUDA Version** | {image_info['cuda_version']} |")
        if image_info.get('pytorch_version'):
            readme_content.append(f"| **PyTorch Version** | {image_info['pytorch_version']} |")
        if image_info.get('tensorflow_version'):
            readme_content.append(f"| **TensorFlow Version** | {image_info['tensorflow_version']} |")
        if analysis.get('python_version'):
            readme_content.append(f"| **Python Version** | {analysis['python_version']} |")
        
        readme_content.append(f"| **Total Packages** | {len(packages)} |")
        readme_content.append("")
        
        # Analysis Results
        readme_content.append("## 🔍 Environment Analysis")
        readme_content.append("")
        readme_content.append("This Docker setup was intelligently generated based on your ComfyUI environment analysis:")
        readme_content.append("")
        readme_content.append("### Detected Components")
        readme_content.append("- **Operating System**: " + report_data.get('system', {}).get('platform', 'Unknown'))
        if report_data.get('system', {}).get('ram'):
            readme_content.append("- **System RAM**: " + report_data['system']['ram'])
        readme_content.append("- **Python Environment**: " + str(analysis.get('python_version') or 'Unknown'))
        readme_content.append("- **GPU Required**: " + ('Yes' if analysis.get('gpu_required') else 'No'))
        if analysis.get('pytorch_detected'):
            readme_content.append("- **PyTorch**: Detected in packages")
        if analysis.get('tensorflow_detected'):
            readme_content.append("- **TensorFlow**: Detected in packages")
        if analysis.get('cuda_version'):
            readme_content.append(f"- **CUDA Version**: {analysis['cuda_version']}")
        readme_content.append("")
        
        # ComfyUI Info
        comfyui_path = comfyui_info.get('path', 'Unknown')
        comfyui_version = comfyui_info.get('version', 'Unknown')
        comfyui_branch = comfyui_info.get('branch', 'master')
        comfyui_remote = comfyui_info.get('remote', 'https://github.com/comfyanonymous/ComfyUI.git')
        
        readme_content.append("### ComfyUI Configuration")
        readme_content.append(f"- **Source Path**: `{comfyui_path}`")
        readme_content.append(f"- **Version**: {comfyui_version}")
        readme_content.append(f"- **Branch**: {comfyui_branch}")
        readme_content.append(f"- **Repository**: {comfyui_remote}")
        readme_content.append("")
        
        # Deployment Options
        readme_content.append("## 🛠️ Deployment Options")
        readme_content.append("")
        readme_content.append("### Option 1: Automated Script (Recommended)")
        readme_content.append("```bash")
        readme_content.append("bash startup.sh")
        readme_content.append("```")
        readme_content.append("This script will:")
        readme_content.append("1. Build the Docker image")
        readme_content.append("2. Create and start the container")
        readme_content.append("3. Copy your models and custom nodes")
        readme_content.append("4. Restart the container to apply changes")
        readme_content.append("")
        
        readme_content.append("### Option 2: Docker Compose")
        readme_content.append("```bash")
        readme_content.append("# Start in background")
        readme_content.append("docker-compose up -d")
        readme_content.append("")
        readme_content.append("# View logs")
        readme_content.append("docker-compose logs -f")
        readme_content.append("")
        readme_content.append("# Stop")
        readme_content.append("docker-compose down")
        readme_content.append("```")
        readme_content.append("")
        
        readme_content.append("### Option 3: Manual Docker Commands")
        readme_content.append("```bash")
        readme_content.append("# Build the image")
        readme_content.append("docker build -t comfyui-custom .")
        readme_content.append("")
        readme_content.append("# Run the container")
        if image_info.get('gpu_support'):
            readme_content.append("docker run -d --name comfyui \\")
            readme_content.append("  -p 8188:8188 \\")
            readme_content.append("  --gpus all \\")
            readme_content.append("  comfyui-custom")
        else:
            readme_content.append("docker run -d --name comfyui \\")
            readme_content.append("  -p 8188:8188 \\")
            readme_content.append("  comfyui-custom")
        readme_content.append("```")
        readme_content.append("")
        
        # File Structure
        readme_content.append("## 📁 File Structure")
        readme_content.append("")
        readme_content.append("```")
        readme_content.append("docker_build_{timestamp}/")
        readme_content.append("├── Dockerfile              # Container definition")
        readme_content.append("├── requirements.txt        # Python packages")
        readme_content.append("├── startup.sh              # Automated setup script")
        readme_content.append("├── docker-compose.yml      # Compose configuration")
        readme_content.append("└── README.md               # This file")
        readme_content.append("```")
        readme_content.append("")
        
        # Dockerfile Explanation
        readme_content.append("## 🐋 Dockerfile Explanation")
        readme_content.append("")
        readme_content.append("The generated Dockerfile is optimized for your specific setup:")
        readme_content.append("")
        readme_content.append("### Base Image Selection")
        readme_content.append(f"**Selected**: `{base_image}`")
        readme_content.append("")
        readme_content.append(f"**Why this image**: {image_info.get('description', 'Custom selection based on your requirements')}")
        readme_content.append("")
        
        # Optimization details
        has_preinstalled_frameworks = any(key in image_info for key in ['pytorch_version', 'tensorflow_version'])
        readme_content.append("### Optimizations Applied")
        if has_preinstalled_frameworks:
            readme_content.append("- ✅ **Framework Pre-installation**: PyTorch/TensorFlow already included")
            readme_content.append("- ✅ **Package Filtering**: Avoids reinstalling pre-built frameworks")
        else:
            readme_content.append("- ✅ **Clean Installation**: All packages installed from requirements.txt")
        
        if image_info.get('gpu_support'):
            readme_content.append("- ✅ **GPU Support**: NVIDIA runtime and CUDA environment configured")
        else:
            readme_content.append("- ✅ **CPU Optimized**: Lightweight setup for CPU-only workloads")
        
        readme_content.append("- ✅ **System Dependencies**: Essential libraries for ComfyUI operation")
        readme_content.append("- ✅ **Port Exposure**: ComfyUI web interface on port 8188")
        readme_content.append("")
        
        # Package List
        readme_content.append("## 📦 Installed Packages")
        readme_content.append("")
        readme_content.append("<details>")
        readme_content.append(f"<summary>View all {len(packages)} packages</summary>")
        readme_content.append("")
        readme_content.append("```")
        for package in packages[:50]:  # Limit to first 50 for readability
            readme_content.append(package)
        if len(packages) > 50:
            readme_content.append(f"... and {len(packages) - 50} more packages")
        readme_content.append("```")
        readme_content.append("")
        readme_content.append("</details>")
        readme_content.append("")
        
        # Troubleshooting
        readme_content.append("## 🔧 Troubleshooting")
        readme_content.append("")
        readme_content.append("### Common Issues")
        readme_content.append("")
        readme_content.append("**Container won't start**")
        readme_content.append("```bash")
        readme_content.append("# Check container logs")
        readme_content.append("docker logs comfyui-container-{timestamp}")
        readme_content.append("")
        readme_content.append("# Check if port is already in use")
        readme_content.append("netstat -tlnp | grep :8188")
        readme_content.append("```")
        readme_content.append("")
        
        if image_info.get('gpu_support'):
            readme_content.append("**GPU not detected**")
            readme_content.append("```bash")
            readme_content.append("# Verify NVIDIA Docker runtime")
            readme_content.append("docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi")
            readme_content.append("")
            readme_content.append("# Check GPU access inside container")
            readme_content.append("docker exec -it comfyui-container-{timestamp} nvidia-smi")
            readme_content.append("```")
            readme_content.append("")
        
        readme_content.append("**Package conflicts**")
        readme_content.append("```bash")
        readme_content.append("# Rebuild with no cache")
        readme_content.append("docker build --no-cache -t comfyui-custom .")
        readme_content.append("")
        readme_content.append("# Check installed packages inside container")
        readme_content.append("docker exec -it comfyui-container-{timestamp} pip list")
        readme_content.append("```")
        readme_content.append("")
        
        # Useful Commands
        readme_content.append("## 💡 Useful Commands")
        readme_content.append("")
        readme_content.append("```bash")
        readme_content.append("# View running containers")
        readme_content.append("docker ps")
        readme_content.append("")
        readme_content.append("# Access container shell")
        readme_content.append("docker exec -it comfyui-container-{timestamp} bash")
        readme_content.append("")
        readme_content.append("# Copy files to/from container")
        readme_content.append("docker cp local_file.txt comfyui-container-{timestamp}:/app/")
        readme_content.append("docker cp comfyui-container-{timestamp}:/app/output/ ./local_output/")
        readme_content.append("")
        readme_content.append("# Monitor resource usage")
        readme_content.append("docker stats comfyui-container-{timestamp}")
        readme_content.append("")
        readme_content.append("# Remove container and image")
        readme_content.append("docker rm -f comfyui-container-{timestamp}")
        readme_content.append("docker rmi comfyui-custom")
        readme_content.append("```")
        readme_content.append("")
        
        # Requirements
        readme_content.append("## 📋 Requirements")
        readme_content.append("")
        readme_content.append("### System Requirements")
        readme_content.append("- **Docker**: Version 20.10 or later")
        if image_info.get('gpu_support'):
            readme_content.append("- **NVIDIA Docker Runtime**: For GPU support")
            readme_content.append("- **NVIDIA Drivers**: Version 470.57.02 or later")
            readme_content.append("- **GPU**: NVIDIA GPU with compute capability 3.5+")
        readme_content.append("- **Memory**: At least 4GB RAM (8GB+ recommended)")
        readme_content.append("- **Storage**: 10GB+ free space for images and models")
        readme_content.append("")
        
        readme_content.append("### Installation Links")
        readme_content.append("- [Docker Desktop](https://docs.docker.com/get-docker/)")
        readme_content.append("- [Docker Compose](https://docs.docker.com/compose/install/)")
        if image_info.get('gpu_support'):
            readme_content.append("- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)")
        readme_content.append("")
        
        # Generation Info
        readme_content.append("## 🤖 Generation Information")
        readme_content.append("")
        readme_content.append("This Docker setup was automatically generated by **ComfyUI Ultimate Inspector Docker Generator**.")
        readme_content.append("")
        readme_content.append("### Generation Details")
        readme_content.append(f"- **Generator Version**: Ultimate Edition v2.0")
        readme_content.append(f"- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        readme_content.append(f"- **Source Report**: Generated from ComfyUI environment analysis")
        readme_content.append(f"- **Optimization Level**: Intelligent image selection with compatibility scoring")
        readme_content.append("")
        readme_content.append("### Customization")
        readme_content.append("Feel free to modify the Dockerfile for your specific needs:")
        readme_content.append("- Add custom environment variables")
        readme_content.append("- Install additional system packages")
        readme_content.append("- Modify Python package versions")
        readme_content.append("- Add custom ComfyUI configurations")
        readme_content.append("")
        
        # Footer
        readme_content.append("---")
        readme_content.append("")
        readme_content.append("🎉 **Happy containerizing!** Your ComfyUI environment is now fully portable and reproducible.")
        readme_content.append("")
        readme_content.append("For issues or improvements, please refer to the ComfyUI Ultimate Inspector documentation.")
        
        # Write README.md
        readme_path = Path(output_dir) / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(readme_content))
        
        print(f"[+] Comprehensive README.md generated: {readme_path}")
        return readme_path
    
    def run(self, base_path="."):
        """Main execution flow"""
        print("🐳 ComfyUI Docker Container Generator")
        print("=" * 50)
        
        # Step 0: Load database
        if not self.database:
            print("[!] Failed to load Docker images database")
            return False
        
        # Step 1: Select report directory
        report_dir = self.select_report_directory(base_path)
        if not report_dir:
            return False
        
        # Parse reports
        print(f"\n[*] Parsing report files...")
        report_data = self.parse_markdown_report(report_dir['md_file'])
        packages = self.load_requirements_file(report_dir['txt_file'])
        
        if not report_data or not packages:
            print("[!] Failed to parse report files")
            return False
        
        print(f"[+] Found {len(packages)} Python packages")
        print(f"[+] ComfyUI path: {report_data.get('comfyui', {}).get('path', 'Unknown')}")
        
        # Step 2: Intelligent Docker base image selection
        base_image, image_info = self.select_docker_base_image(report_data, packages)
        if not base_image:
            return False
        
        # Extract GPU requirement and analysis for later use
        gpu_required = report_data.get('gpu_info', {}).get('has_cuda', False)
        analysis = self.analyze_report_requirements(report_data, packages)
        
        # Step 3: Create output directory
        output_dir = Path(f"docker_build_{report_dir['timestamp']}")
        output_dir.mkdir(exist_ok=True)
        print(f"\n[*] Creating Docker files in: {output_dir}")
        
        # Copy requirements.txt
        shutil.copy2(report_dir['txt_file'], output_dir / "requirements.txt")
        print(f"[+] Copied requirements.txt")
        
        # Step 4: Generate optimized Dockerfile
        dockerfile_path = self.generate_dockerfile(
            base_image, image_info, packages, report_data.get('comfyui', {}), output_dir
        )
        
        # Step 5: Generate startup script
        startup_script = self.generate_startup_script(
            report_data.get('comfyui', {}).get('path'),
            {**report_data.get('models_info', {}), 'gpu_required': gpu_required},
            output_dir
        )
        
        # Step 6: Generate docker-compose.yml
        try:
            compose_file = self.generate_docker_compose(
                {'gpu_required': gpu_required}, output_dir
            )
        except ImportError:
            print("[!] PyYAML not installed, skipping docker-compose.yml generation")
            print("    Install with: pip install pyyaml")
        
        # Step 7: Generate comprehensive README.md
        readme_file = self.generate_readme(
            base_image, image_info, packages, report_data.get('comfyui', {}),
            {**report_data.get('models_info', {}), 'gpu_required': gpu_required},
            report_data, analysis, output_dir
        )
        
        # Summary
        print(f"\n🎉 Docker generation complete!")
        print(f"📁 Output directory: {output_dir}")
        print(f"🐳 Base image: {base_image}")
        print(f"📦 Python packages: {len(packages)}")
        print(f"🎮 GPU support: {'Yes' if gpu_required else 'No'}")
        print(f"\n📋 Next steps:")
        print(f"   1. cd {output_dir}")
        print(f"   2. Run: bash startup.sh")
        print(f"   3. Open: http://localhost:8188")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Generate Docker containers from ComfyUI reports')
    parser.add_argument('--path', '-p', default='.', help='Base path to search for reports (default: current directory)')
    parser.add_argument('--database', '-d', default='docker_images_db.json', help='Docker images database file')
    
    args = parser.parse_args()
    
    generator = ComfyUIDockerGenerator(args.database)
    success = generator.run(args.path)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()