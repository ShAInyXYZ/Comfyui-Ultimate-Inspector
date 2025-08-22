"""
ComfyUI Ultimate Inspector
Comprehensive deep-dive analysis of ComfyUI installation and environment
"""

import os
import sys
import json
import subprocess
import platform
import re
import yaml
import importlib.util
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import argparse
import tempfile

def run_python_command(python_path, command, timeout=30):
    """Run a Python command and return output safely"""
    try:
        result = subprocess.run([str(python_path), "-c", command], 
                              capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), -1

def detect_gpu_info(python_path):
    """Detect detailed GPU information including CUDA/PyTorch info"""
    gpu_info = {}
    
    # Get NVIDIA driver version
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version,memory.total,name', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        gpus.append({
                            'name': parts[2],
                            'driver_version': parts[0],
                            'memory_mb': parts[1],
                            'memory_gb': f"{float(parts[1]) / 1024:.1f}"
                        })
            gpu_info['nvidia_gpus'] = gpus
    except:
        gpu_info['nvidia_gpus'] = []
    
    # Get CUDA info via PyTorch if available
    cuda_cmd = """
import sys
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"GPU {i} memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"GPU {i} compute: {props.major}.{props.minor}")
except ImportError:
    print("PyTorch not available")
except Exception as e:
    print(f"Error: {e}")
"""
    
    stdout, stderr, code = run_python_command(python_path, cuda_cmd)
    gpu_info['pytorch_cuda_info'] = stdout
    if stderr:
        gpu_info['pytorch_cuda_errors'] = stderr
    
    return gpu_info

def detect_python_environment_type(python_path):
    """Detect if Python is in venv, conda, embedded, etc."""
    env_info = {}
    
    # Check if it's in a virtual environment
    venv_cmd = """
import sys
import os
print(f"sys.prefix: {sys.prefix}")
print(f"sys.base_prefix: {sys.base_prefix}")
print(f"hasattr sys, 'real_prefix': {hasattr(sys, 'real_prefix')}")
print(f"sys.prefix != sys.base_prefix: {sys.prefix != sys.base_prefix}")

# Check for conda
if 'conda' in sys.prefix or 'Anaconda' in sys.prefix or 'Miniconda' in sys.prefix:
    print("Environment type: conda")
elif hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    print("Environment type: virtual environment")
elif 'embedded' in sys.prefix.lower() or 'embeded' in sys.prefix.lower():
    print("Environment type: embedded")
else:
    print("Environment type: system")

# Additional conda check
try:
    import conda
    print("Conda package available")
except ImportError:
    pass
"""
    
    stdout, stderr, code = run_python_command(python_path, venv_cmd)
    env_info['detection_output'] = stdout
    
    # Parse the output
    lines = stdout.split('\n')
    for line in lines:
        if 'Environment type:' in line:
            env_info['type'] = line.split(': ')[1]
            break
    else:
        env_info['type'] = 'unknown'
    
    # Check for conda environment name
    try:
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        if conda_env:
            env_info['conda_env_name'] = conda_env
    except:
        pass
    
    return env_info

def get_full_pip_freeze(python_path, output_dir):
    """Generate organized pip list output and save to file with timestamp"""
    freeze_info = {}
    
    try:
        # Use pip list instead of pip freeze for consistency with markdown
        result = subprocess.run([str(python_path), "-m", "pip", "list"], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            # Parse pip list output (same logic as get_installed_packages_detailed)
            lines = result.stdout.split('\n')[2:]  # Skip header
            valid_packages = []
            
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        name = parts[0]
                        version = parts[1]
                        # Format as package==version for requirements.txt compatibility
                        valid_packages.append(f"{name}=={version}")
            
            # Sort packages alphabetically
            valid_packages.sort(key=lambda x: x.lower())
            
            freeze_info['package_count'] = len(valid_packages)
            
            # Generate filename with timestamp: requirements_(Day-HourMinute).txt
            now = datetime.now()
            timestamp = now.strftime("%d-%H%M")
            freeze_file = Path(output_dir) / f"requirements_{timestamp}.txt"
            
            with open(freeze_file, 'w', encoding='utf-8') as f:
                f.write(f"# Generated pip list output (requirements format)\n")
                f.write(f"# Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Python: {python_path}\n")
                f.write(f"# Total packages: {len(valid_packages)}\n\n")
                
                # Write all packages in requirements.txt format
                for req in valid_packages:
                    f.write(f"{req}\n")
            
            freeze_info['file_path'] = str(freeze_file)
            freeze_info['success'] = True
        else:
            freeze_info['error'] = result.stderr
            freeze_info['success'] = False
    except Exception as e:
        freeze_info['error'] = str(e)
        freeze_info['success'] = False
    
    return freeze_info

def parse_requirements(req_file):
    """Enhanced requirements parsing with version constraints"""
    requirements = []
    try:
        with open(req_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-'):
                    # Handle git URLs and other complex requirements
                    if line.startswith('git+'):
                        requirements.append({'raw': line, 'type': 'git', 'line': line_num})
                    elif '://' in line:
                        requirements.append({'raw': line, 'type': 'url', 'line': line_num})
                    else:
                        # Regular package requirement
                        req = line.split()[0].split(';')[0]  # Remove comments and conditions
                        requirements.append({'raw': req, 'type': 'package', 'line': line_num})
    except Exception as e:
        requirements.append({'raw': f'Error reading {req_file}: {e}', 'type': 'error', 'line': 0})
    return requirements

def parse_package_requirement(req_str):
    """Enhanced package parsing with better version constraint handling"""
    # Handle various formats: package==1.0.0, package>=1.0, package~=1.0, etc.
    operators = ['==', '>=', '<=', '>', '<', '~=', '!=']
    
    package_name = req_str
    constraints = []
    
    for op in operators:
        if op in req_str:
            parts = req_str.split(op)
            if len(parts) == 2:
                package_name = parts[0].strip()
                version = parts[1].strip()
                constraints.append({'operator': op, 'version': version})
    
    return {
        'name': package_name.lower().replace('-', '_'),
        'original_name': package_name,
        'constraints': constraints,
        'raw': req_str
    }

def check_comfyui_logs(comfy_path, max_lines=100):
    """Check for ComfyUI logs and extract errors/warnings"""
    log_info = {}
    
    # Common log locations
    log_locations = [
        comfy_path / "comfyui.log",
        comfy_path / "logs" / "comfyui.log",
        comfy_path / "output.log",
        comfy_path / "error.log"
    ]
    
    # Also check for recent console output files
    for pattern in ["*.log", "*.out", "*.err"]:
        log_locations.extend(comfy_path.glob(pattern))
    
    found_logs = []
    errors_warnings = []
    
    for log_path in log_locations:
        if log_path.exists() and log_path.is_file():
            try:
                # Get file info
                stat = log_path.stat()
                found_logs.append({
                    'path': str(log_path),
                    'size_kb': round(stat.st_size / 1024, 2),
                    'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                })
                
                # Read recent lines and look for errors/warnings
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    recent_lines = lines[-max_lines:] if len(lines) > max_lines else lines
                    
                    for line_num, line in enumerate(recent_lines):
                        line_lower = line.lower()
                        if any(keyword in line_lower for keyword in ['error', 'warning', 'exception', 'traceback', 'failed']):
                            errors_warnings.append({
                                'file': log_path.name,
                                'line': line.strip(),
                                'line_number': len(lines) - len(recent_lines) + line_num + 1
                            })
            except Exception as e:
                found_logs.append({
                    'path': str(log_path),
                    'error': f"Could not read: {e}"
                })
    
    log_info['found_logs'] = found_logs
    log_info['errors_warnings'] = errors_warnings[:50]  # Limit to 50 most recent
    
    return log_info

def check_extra_model_paths(comfy_path):
    """Check for extra_model_paths.yaml configuration"""
    config_info = {}
    
    config_file = comfy_path / "extra_model_paths.yaml"
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_content = f.read()
                config_info['exists'] = True
                config_info['size_kb'] = round(len(config_content) / 1024, 2)
                
                # Try to parse YAML
                try:
                    config_data = yaml.safe_load(config_content)
                    if config_data:
                        config_info['sections'] = list(config_data.keys())
                        config_info['total_paths'] = sum(len(v) if isinstance(v, dict) else 1 for v in config_data.values())
                    else:
                        config_info['empty'] = True
                except yaml.YAMLError as e:
                    config_info['yaml_error'] = str(e)
                    
        except Exception as e:
            config_info['exists'] = True
            config_info['error'] = str(e)
    else:
        config_info['exists'] = False
    
    return config_info

def parse_extra_model_paths(comfy_path):
    """Parse extra_model_paths.yaml and return all configured paths"""
    paths_data = {}
    
    config_file = comfy_path / "extra_model_paths.yaml"
    if not config_file.exists():
        return paths_data
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            
        if not config_data:
            return paths_data
        
        # Extract paths from each configuration section
        for section_name, section_config in config_data.items():
            if not isinstance(section_config, dict):
                continue
                
            base_path = section_config.get('base_path', '')
            if not base_path:
                continue
                
            section_paths = {}
            for model_type, model_paths in section_config.items():
                if model_type == 'base_path' or model_type == 'is_default':
                    continue
                    
                # Handle single path or multiple paths
                if isinstance(model_paths, str):
                    # Single path - combine with base_path if relative
                    if model_paths.strip().startswith(('/', 'X:', 'C:', 'D:')):
                        # Absolute path
                        section_paths[model_type] = [model_paths.strip()]
                    else:
                        # Relative path
                        section_paths[model_type] = [str(Path(base_path) / model_paths.strip())]
                elif isinstance(model_paths, list):
                    # List of paths
                    resolved_paths = []
                    for path in model_paths:
                        path = path.strip()
                        if path.startswith(('/', 'X:', 'C:', 'D:')):
                            resolved_paths.append(path)
                        else:
                            resolved_paths.append(str(Path(base_path) / path))
                    section_paths[model_type] = resolved_paths
                else:
                    # Multi-line string format
                    path_lines = str(model_paths).strip().split('\n')
                    resolved_paths = []
                    for line in path_lines:
                        path = line.strip()
                        if not path:
                            continue
                        if path.startswith(('/', 'X:', 'C:', 'D:')):
                            resolved_paths.append(path)
                        else:
                            resolved_paths.append(str(Path(base_path) / path))
                    section_paths[model_type] = resolved_paths
            
            if section_paths:
                paths_data[section_name] = section_paths
                
    except Exception as e:
        paths_data['error'] = str(e)
    
    return paths_data

def scan_model_directory(dir_path, model_type):
    """Scan a directory for model files and return details"""
    result = {
        'exists': False,
        'count': 0,
        'total_size_gb': 0.0,
        'models': []
    }
    
    try:
        dir_path = Path(dir_path)
        if not dir_path.exists():
            return result
            
        result['exists'] = True
        
        # Common model file extensions
        model_extensions = {'.safetensors', '.ckpt', '.pt', '.pth', '.bin', '.gguf'}
        
        total_size = 0
        models = []
        
        for file_path in dir_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in model_extensions:
                try:
                    size = file_path.stat().st_size
                    total_size += size
                    
                    models.append({
                        'name': file_path.name,
                        'size_gb': round(size / (1024**3), 2),
                        'extension': file_path.suffix,
                        'modified': datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d'),
                        'path': str(file_path.relative_to(dir_path)) if len(str(file_path.relative_to(dir_path))) < 50 else file_path.name
                    })
                except:
                    continue
        
        result['count'] = len(models)
        result['total_size_gb'] = round(total_size / (1024**3), 2)
        result['models'] = sorted(models, key=lambda x: x['size_gb'], reverse=True)
        
    except Exception as e:
        result['error'] = str(e)
    
    return result

def get_model_inventory(comfy_path):
    """Get comprehensive model inventory from both default and extra paths"""
    models_info = {}
    
    # Default ComfyUI model directories
    models_dir = comfy_path / "models"
    default_model_types = [
        'checkpoints', 'loras', 'vae', 'vae_approx', 'clip', 'clip_vision', 
        'text_encoders', 'controlnet', 'diffusion_models', 'unet', 'diffusers',
        'upscale_models', 'embeddings', 'hypernetworks', 'gligen', 
        'style_models', 'photomaker', 'pulid'
    ]
    
    # Scan default directories
    for model_type in default_model_types:
        type_dir = models_dir / model_type
        models_info[f"default_{model_type}"] = scan_model_directory(type_dir, model_type)
    
    # Parse and scan extra model paths
    extra_paths = parse_extra_model_paths(comfy_path)
    for section_name, section_paths in extra_paths.items():
        if section_name == 'error':
            continue
            
        for model_type, paths in section_paths.items():
            for i, path in enumerate(paths):
                key = f"{section_name}_{model_type}"
                if i > 0:
                    key += f"_{i+1}"
                models_info[key] = scan_model_directory(path, model_type)
    
    return models_info

def get_comfyui_manager_info(comfy_path):
    """Get ComfyUI-Manager information and statistics"""
    manager_info = {}
    manager_path = comfy_path / "custom_nodes" / "ComfyUI-Manager"
    
    manager_info['path'] = str(manager_path)
    manager_info['exists'] = manager_path.exists()
    
    if not manager_path.exists():
        manager_info['status'] = 'Not installed'
        return manager_info
    
    manager_info['status'] = 'Installed'
    
    # Get version from pyproject.toml
    pyproject_file = manager_path / "pyproject.toml"
    if pyproject_file.exists():
        try:
            with open(pyproject_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Extract version
                for line in content.split('\n'):
                    if line.strip().startswith('version ='):
                        version = line.split('=')[1].strip().strip('"\'')
                        manager_info['version'] = version
                        break
        except Exception as e:
            manager_info['version_error'] = str(e)
    
    # Get git information if available
    git_dir = manager_path / ".git"
    if git_dir.exists():
        try:
            # Get last commit info
            result = subprocess.run(["git", "-C", str(manager_path), "log", "-1", "--format=%cd|%s", "--date=short"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and '|' in result.stdout:
                date, message = result.stdout.strip().split('|', 1)
                manager_info['last_update'] = date
                manager_info['last_commit_message'] = message
        except Exception as e:
            manager_info['git_error'] = str(e)
    
    # Database statistics
    databases = {
        'custom_nodes': manager_path / "custom-node-list.json",
        'models': manager_path / "model-list.json",
        'extensions': manager_path / "extension-node-map.json"
    }
    
    manager_info['databases'] = {}
    for db_name, db_path in databases.items():
        if db_path.exists():
            try:
                stat = db_path.stat()
                size_kb = round(stat.st_size / 1024, 1)
                modified = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d')
                
                # Count entries for JSON files
                entry_count = None
                if db_path.suffix == '.json':
                    try:
                        with open(db_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, dict):
                                # Look for common array keys
                                for key in ['custom_nodes', 'models', 'extensions']:
                                    if key in data and isinstance(data[key], list):
                                        entry_count = len(data[key])
                                        break
                                if entry_count is None and 'models' in data:
                                    entry_count = len(data['models'])
                            elif isinstance(data, list):
                                entry_count = len(data)
                    except:
                        pass
                
                manager_info['databases'][db_name] = {
                    'size_kb': size_kb,
                    'modified': modified,
                    'entries': entry_count
                }
            except Exception as e:
                manager_info['databases'][db_name] = {'error': str(e)}
        else:
            manager_info['databases'][db_name] = {'exists': False}
    
    # Check for snapshots
    snapshots_dir = manager_path / "snapshots"
    if snapshots_dir.exists():
        try:
            snapshot_files = list(snapshots_dir.glob("*.json"))
            manager_info['snapshots_count'] = len(snapshot_files)
        except:
            manager_info['snapshots_count'] = 0
    else:
        manager_info['snapshots_count'] = 0
    
    # Check key features/files
    key_files = {
        'web_interface': manager_path / "js" / "comfyui-manager.js",
        'cli_tool': manager_path / "cm-cli.py",
        'security_check': manager_path / "glob" / "security_check.py"
    }
    
    manager_info['features'] = {}
    for feature, file_path in key_files.items():
        manager_info['features'][feature] = file_path.exists()
    
    return manager_info

def find_python_executable(root_path):
    """Find embedded Python executable in ComfyUI installation"""
    candidates = [
        root_path / "python_embeded" / "python.exe",
        root_path / "python_embedded" / "python.exe", 
        root_path / "python" / "python.exe",
        root_path / "Python" / "python.exe",
        root_path / "python_embeded" / "python",
        root_path / "python_embedded" / "python",
        root_path / "python" / "python",
    ]
    
    # Check candidates first
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    # Recursive search as fallback
    for python_exe in root_path.rglob("python*"):
        if python_exe.is_file() and python_exe.name in ["python.exe", "python"]:
            return python_exe
    
    return None

def get_python_info_enhanced(python_path, output_dir):
    """Get comprehensive Python environment information"""
    info = {}
    
    try:
        # Basic version info
        result = subprocess.run([str(python_path), "-V"], 
                              capture_output=True, text=True, timeout=10)
        info['version'] = result.stdout.strip() if result.stdout else result.stderr.strip()
        
        # Full Python info
        result = subprocess.run([str(python_path), "-c", 
                               "import sys; print(sys.version.replace('\\n', ' '))"], 
                              capture_output=True, text=True, timeout=10)
        info['full_version'] = result.stdout.strip()
        
        # Environment type detection
        info['environment'] = detect_python_environment_type(python_path)
        
        # Get installed packages with versions
        info['packages'] = get_installed_packages_detailed(python_path)
        
        # Full pip freeze
        info['pip_freeze'] = get_full_pip_freeze(python_path, output_dir)
        
        # GPU/CUDA information
        info['gpu_info'] = detect_gpu_info(python_path)
        
    except Exception as e:
        info['error'] = str(e)
    
    return info

def get_installed_packages_detailed(python_path):
    """Get detailed installed packages with versions"""
    packages = {}
    
    try:
        # Try pip list first
        result = subprocess.run([str(python_path), "-m", "pip", "list"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            lines = result.stdout.split('\n')[2:]  # Skip header
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        name = parts[0].lower().replace('-', '_')
                        version = parts[1]
                        packages[name] = version
        else:
            # Fallback: parse site-packages manually
            site_packages = python_path.parent / "Lib" / "site-packages"
            if not site_packages.exists():
                site_packages = python_path.parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
            
            if site_packages.exists():
                for dist_info in site_packages.glob("*.dist-info"):
                    name_version = dist_info.name.replace('.dist-info', '')
                    parts = name_version.split('-')
                    if len(parts) >= 2:
                        name = '-'.join(parts[:-1]).lower().replace('-', '_')
                        version = parts[-1]
                        packages[name] = version
    
    except Exception as e:
        packages['error'] = str(e)
    
    return packages

def get_comfyui_info_enhanced(root_path):
    """Get enhanced ComfyUI version and configuration information"""
    info = {}
    comfy_path = root_path / "ComfyUI"
    
    info['path'] = str(comfy_path)
    info['exists'] = comfy_path.exists()
    
    if comfy_path.exists():
        # Git information
        git_dir = comfy_path / ".git"
        if git_dir.exists():
            try:
                # Get full commit hash
                result = subprocess.run(["git", "-C", str(comfy_path), "rev-parse", "HEAD"], 
                                      capture_output=True, text=True, timeout=10)
                full_commit = result.stdout.strip() if result.returncode == 0 else "unknown"
                
                # Get short commit hash
                result = subprocess.run(["git", "-C", str(comfy_path), "rev-parse", "--short", "HEAD"], 
                                      capture_output=True, text=True, timeout=10)
                short_commit = result.stdout.strip() if result.returncode == 0 else "unknown"
                
                # Get branch
                result = subprocess.run(["git", "-C", str(comfy_path), "rev-parse", "--abbrev-ref", "HEAD"], 
                                      capture_output=True, text=True, timeout=10)
                branch = result.stdout.strip() if result.returncode == 0 else "unknown"
                
                # Get last commit date and message
                result = subprocess.run(["git", "-C", str(comfy_path), "log", "-1", "--format=%cd|%s", "--date=short"], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0 and '|' in result.stdout:
                    date, message = result.stdout.strip().split('|', 1)
                    info['last_commit_date'] = date
                    info['last_commit_message'] = message
                
                # Get remote URL
                result = subprocess.run(["git", "-C", str(comfy_path), "remote", "get-url", "origin"], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    info['git_remote'] = result.stdout.strip()
                
                info['version'] = f"{short_commit} ({branch})"
                info['full_commit'] = full_commit
                info['short_commit'] = short_commit
                info['branch'] = branch
                
            except Exception as e:
                info['version'] = f"Git error: {e}"
        else:
            info['version'] = "No git repository"
            
        # Check for main ComfyUI files
        main_files = ['main.py', 'server.py', 'execution.py', 'nodes.py', 'model_management.py']
        info['core_files'] = {}
        for file in main_files:
            file_path = comfy_path / file
            info['core_files'][file] = file_path.exists()
        
        # Check configuration files
        info['config'] = check_extra_model_paths(comfy_path)
        
        # Check logs
        info['logs'] = check_comfyui_logs(comfy_path)
        
        # Get model inventory
        info['models'] = get_model_inventory(comfy_path)
        
        # Get ComfyUI-Manager information
        info['manager'] = get_comfyui_manager_info(comfy_path)
    else:
        info['version'] = "ComfyUI directory not found"
    
    return info

def analyze_custom_node_enhanced(node_path, installed_packages):
    """Enhanced custom node analysis with detailed Git and dependency info"""
    analysis = {
        'name': node_path.name,
        'path': str(node_path),
        'requirements': [],
        'requirement_files': [],
        'missing_packages': [],
        'version_conflicts': [],
        'has_requirements_file': False,
        'has_install_script': False,
        'python_files': 0,
        'git_info': {},
        'size_mb': 0,
        'file_stats': {}
    }
    
    # Enhanced git info
    git_dir = node_path / ".git"
    if git_dir.exists():
        try:
            # Get remote URL
            result = subprocess.run(["git", "-C", str(node_path), "remote", "get-url", "origin"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                analysis['git_info']['url'] = result.stdout.strip()
            
            # Get full commit hash
            result = subprocess.run(["git", "-C", str(node_path), "rev-parse", "HEAD"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                analysis['git_info']['commit_hash'] = result.stdout.strip()
                analysis['git_info']['short_hash'] = result.stdout.strip()[:8]
            
            # Get last commit date and message
            result = subprocess.run(["git", "-C", str(node_path), "log", "-1", "--format=%cd|%s", "--date=short"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and '|' in result.stdout:
                date, message = result.stdout.strip().split('|', 1)
                analysis['git_info']['last_commit_date'] = date
                analysis['git_info']['last_commit_message'] = message
            
            # Get branch
            result = subprocess.run(["git", "-C", str(node_path), "rev-parse", "--abbrev-ref", "HEAD"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                analysis['git_info']['branch'] = result.stdout.strip()
        except:
            pass
    
    # File statistics
    try:
        file_types = defaultdict(int)
        total_size = 0
        
        for file_path in node_path.rglob('*'):
            if file_path.is_file():
                try:
                    size = file_path.stat().st_size
                    total_size += size
                    suffix = file_path.suffix.lower()
                    file_types[suffix] += 1
                except:
                    pass
        
        analysis['size_mb'] = round(total_size / (1024 * 1024), 2)
        analysis['file_stats'] = dict(file_types)
        analysis['python_files'] = file_types.get('.py', 0)
    except:
        pass
    
    # Enhanced requirements analysis
    req_files = ['requirements.txt', 'requirements.pip', 'requirements-dev.txt', 'pyproject.toml', 'setup.py']
    for req_file in req_files:
        req_path = node_path / req_file
        if req_path.exists():
            analysis['has_requirements_file'] = True
            analysis['requirement_files'].append(req_file)
            
            if req_file.endswith('.txt') or req_file.endswith('.pip'):
                requirements = parse_requirements(req_path)
                analysis['requirements'].extend(requirements)
    
    # Check for install scripts
    install_scripts = ['install.py', 'setup.py', 'install.bat', 'install.sh', 'requirements.sh']
    for script in install_scripts:
        if (node_path / script).exists():
            analysis['has_install_script'] = True
            break
    
    # Enhanced dependency analysis
    if isinstance(installed_packages, dict):
        for req in analysis['requirements']:
            if req['type'] == 'package':
                parsed = parse_package_requirement(req['raw'])
                package_name = parsed['name']
                
                if package_name not in installed_packages:
                    analysis['missing_packages'].append({
                        'package': req['raw'],
                        'parsed_name': package_name,
                        'source_file': 'requirements.txt'  # Could be enhanced to track source
                    })
                elif parsed['constraints']:
                    # Check version constraints
                    installed_version = installed_packages[package_name]
                    for constraint in parsed['constraints']:
                        compatible = compare_versions(installed_version, constraint['version'], constraint['operator'])
                        if compatible is False:
                            analysis['version_conflicts'].append({
                                'package': package_name,
                                'required': f"{constraint['operator']}{constraint['version']}",
                                'installed': installed_version,
                                'constraint_type': constraint['operator']
                            })
    
    return analysis

def compare_versions(installed_version, required_version, operator='>='):
    """Enhanced version comparison with better handling of different formats"""
    try:
        def normalize_version(v):
            # Handle versions like "1.0.0+cpu" or "2.0.0.dev0"
            v = re.split(r'[\+\-]', str(v))[0]  # Remove suffixes
            v = re.sub(r'[a-zA-Z]', '', v)  # Remove letters
            parts = [int(x) for x in v.split('.') if x.isdigit()]
            return parts
        
        if not installed_version or not required_version:
            return None
        
        installed = normalize_version(installed_version)
        required = normalize_version(required_version)
        
        # Pad shorter version with zeros
        max_len = max(len(installed), len(required))
        installed.extend([0] * (max_len - len(installed)))
        required.extend([0] * (max_len - len(required)))
        
        if operator == '>=':
            return installed >= required
        elif operator == '==':
            return installed == required
        elif operator == '>':
            return installed > required
        elif operator == '<=':
            return installed <= required
        elif operator == '<':
            return installed < required
        elif operator == '~=':
            # Compatible release operator
            return installed >= required and installed[0] == required[0]
        elif operator == '!=':
            return installed != required
        else:
            return None
    except:
        return None

def get_custom_nodes_enhanced(root_path, installed_packages):
    """Get enhanced analysis of custom nodes"""
    nodes = []
    custom_nodes_path = root_path / "ComfyUI" / "custom_nodes"
    
    if custom_nodes_path.exists():
        for item in custom_nodes_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                analysis = analyze_custom_node_enhanced(item, installed_packages)
                nodes.append(analysis)
    
    return sorted(nodes, key=lambda x: x['name'].lower())

def generate_dependency_lock_summary(nodes):
    """Generate consolidated dependency analysis across all nodes"""
    summary = {
        'all_requirements': defaultdict(list),
        'conflicting_requirements': {},
        'missing_by_node': {},
        'most_common': [],
        'git_requirements': [],
        'problematic_packages': []
    }
    
    # Collect all requirements
    for node in nodes:
        node_name = node['name']
        
        # Track missing packages
        if node['missing_packages']:
            summary['missing_by_node'][node_name] = node['missing_packages']
        
        for req in node['requirements']:
            if req['type'] == 'package':
                parsed = parse_package_requirement(req['raw'])
                package_name = parsed['name']
                summary['all_requirements'][package_name].append({
                    'node': node_name,
                    'requirement': req['raw'],
                    'constraints': parsed['constraints']
                })
            elif req['type'] == 'git':
                summary['git_requirements'].append({
                    'node': node_name,
                    'requirement': req['raw']
                })
    
    # Find conflicting requirements
    for package, requirements in summary['all_requirements'].items():
        if len(requirements) > 1:
            # Check for different version constraints
            constraints = []
            for req in requirements:
                for constraint in req['constraints']:
                    constraints.append(f"{constraint['operator']}{constraint['version']}")
            
            unique_constraints = set(constraints)
            if len(unique_constraints) > 1:
                summary['conflicting_requirements'][package] = {
                    'requirements': requirements,
                    'constraints': list(unique_constraints)
                }
    
    # Most common packages
    package_counts = Counter(summary['all_requirements'].keys())
    summary['most_common'] = package_counts.most_common(20)
    
    return summary

def get_system_info_enhanced():
    """Get enhanced system hardware and OS information"""
    info = {}
    
    # Basic system info
    info['platform'] = platform.platform()
    info['machine'] = platform.machine()
    info['processor'] = platform.processor()
    info['python_version'] = platform.python_version()
    info['architecture'] = platform.architecture()
    
    # Try to get more detailed info based on OS
    if platform.system() == "Windows":
        info.update(get_windows_info_enhanced())
    elif platform.system() == "Linux":
        info.update(get_linux_info_enhanced())
    elif platform.system() == "Darwin":
        info.update(get_macos_info_enhanced())
    
    return info

def get_windows_info_enhanced():
    """Get enhanced Windows-specific system information"""
    info = {}
    try:
        # Get detailed system info using wmic
        queries = [
            ('cpu_info', ['wmic', 'cpu', 'get', 'Name,NumberOfCores,NumberOfLogicalProcessors', '/format:csv']),
            ('ram_info', ['wmic', 'computersystem', 'get', 'TotalPhysicalMemory', '/value']),
            ('gpu_info', ['wmic', 'path', 'win32_VideoController', 'get', 'Name,AdapterRAM', '/format:csv']),
            ('os_info', ['wmic', 'os', 'get', 'Caption,Version,BuildNumber', '/format:csv'])
        ]
        
        for key, command in queries:
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=15)
                if result.returncode == 0:
                    info[key] = result.stdout.strip()
            except:
                pass
        
        # Parse RAM info
        if 'ram_info' in info:
            for line in info['ram_info'].split('\n'):
                if 'TotalPhysicalMemory' in line and '=' in line:
                    try:
                        ram_bytes = int(line.split('=')[1].strip())
                        info['ram_gb'] = round(ram_bytes / (1024**3), 1)
                    except:
                        pass
        
    except Exception as e:
        info['error'] = f"Windows info error: {e}"
    
    return info

def get_linux_info_enhanced():
    """Get enhanced Linux-specific system information"""
    info = {}
    try:
        # CPU info
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpu_info = f.read()
                info['cpu_info'] = cpu_info[:500]  # First 500 chars
        except:
            pass
        
        # Memory info
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        ram_kb = int(line.split()[1])
                        info['ram_gb'] = round(ram_kb / (1024**2), 1)
                        break
        except:
            pass
        
        # GPU info
        try:
            result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.split('\n'):
                    if 'VGA' in line or 'Display' in line or '3D' in line:
                        gpus.append(line.split(': ')[-1] if ': ' in line else line)
                info['gpus'] = gpus
        except:
            pass
    
    except Exception as e:
        info['error'] = f"Linux info error: {e}"
    
    return info

def get_macos_info_enhanced():
    """Get enhanced macOS-specific system information"""
    info = {}
    try:
        # Get system info
        result = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                              capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            info['hardware_info'] = result.stdout[:1000]  # First 1000 chars
            for line in result.stdout.split('\n'):
                if 'Memory:' in line:
                    info['ram_info'] = line.strip()
    
    except Exception as e:
        info['error'] = f"macOS info error: {e}"
    
    return info

def generate_ultimate_markdown_report(data, output_dir):
    """Generate the ultimate comprehensive markdown report"""
    md = []
    
    # Header
    md.append("# 🚀 ComfyUI Ultimate System Analysis Report")
    md.append("")
    md.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"**Root Path:** `{data['root_path']}`")
    md.append(f"**Analysis Version:** Ultimate Edition v2.0")
    md.append("")
    md.append("---")
    md.append("")
    
    # Table of Contents
    md.append("## 📑 Table of Contents")
    md.append("1. [🔧 System Information](#-system-information)")
    md.append("2. [🎨 ComfyUI Installation](#-comfyui-installation)")
    md.append("3. [🐍 Python Environment](#-python-environment)")
    md.append("4. [🔧 Custom Nodes Deep Dive](#-custom-nodes-deep-dive)")
    md.append("5. [📦 Dependency Lock Summary](#-dependency-lock-summary)")
    md.append("6. [⚙️ Configuration & Runtime](#-configuration--runtime)")
    md.append("7. [💡 Recommendations](#-recommendations)")
    md.append("")
    md.append("---")
    md.append("")
    
    # System Information with GPU details
    md.append("## 🔧 System Information")
    md.append("")
    system = data['system']
    md.append(f"- **Platform:** {system['platform']}")
    md.append(f"- **Architecture:** {system['machine']} / {system.get('architecture', ['Unknown'])[0]}")
    md.append(f"- **Processor:** {system.get('processor', 'Unknown')}")
    if 'ram_gb' in system:
        md.append(f"- **RAM:** {system['ram_gb']} GB")
    
    # GPU Information
    python_info = data.get('python', {})
    gpu_info = python_info.get('gpu_info', {})
    
    if gpu_info.get('nvidia_gpus'):
        md.append("- **NVIDIA GPUs:**")
        for gpu in gpu_info['nvidia_gpus']:
            md.append(f"  - **{gpu['name']}** (Driver: {gpu['driver_version']}, VRAM: {gpu['memory_gb']} GB)")
    
    if gpu_info.get('pytorch_cuda_info'):
        md.append("- **PyTorch CUDA Info:**")
        for line in gpu_info['pytorch_cuda_info'].split('\n')[:5]:  # First 5 lines
            if line.strip():
                md.append(f"  - {line.strip()}")
    
    md.append("")
    
    # ComfyUI Installation Enhanced
    md.append("## 🎨 ComfyUI Installation")
    md.append("")
    comfy = data['comfyui']
    md.append(f"- **Path:** `{comfy['path']}`")
    md.append(f"- **Exists:** {'✅ Yes' if comfy['exists'] else '❌ No'}")
    md.append(f"- **Version:** {comfy['version']}")
    
    if 'full_commit' in comfy:
        md.append(f"- **Full Commit Hash:** `{comfy['full_commit']}`")
    if 'branch' in comfy:
        md.append(f"- **Branch:** {comfy['branch']}")
    if 'git_remote' in comfy:
        md.append(f"- **Remote:** {comfy['git_remote']}")
    if 'last_commit_date' in comfy:
        md.append(f"- **Last Update:** {comfy['last_commit_date']}")
    if 'last_commit_message' in comfy:
        md.append(f"- **Last Commit:** {comfy['last_commit_message']}")
    
    if 'core_files' in comfy:
        md.append("- **Core Files:**")
        for file, exists in comfy['core_files'].items():
            md.append(f"  - {file}: {'✅' if exists else '❌'}")
    md.append("")
    
    # Python Environment Enhanced
    md.append("## 🐍 Python Environment")
    md.append("")
    if 'python' in data and 'error' not in data['python']:
        python = data['python']
        md.append(f"- **Path:** `{python['path']}`")
        md.append(f"- **Version:** {python['version']}")
        md.append(f"- **Full Info:** {python.get('full_version', 'N/A')}")
        
        # Environment type
        env_info = python.get('environment', {})
        md.append(f"- **Environment Type:** {env_info.get('type', 'Unknown')}")
        if 'conda_env_name' in env_info:
            md.append(f"- **Conda Environment:** {env_info['conda_env_name']}")
        
        # Package info
        if isinstance(python['packages'], dict):
            md.append(f"- **Installed Packages:** {len(python['packages'])}")
        
        # Pip freeze info
        pip_freeze = python.get('pip_freeze', {})
        if pip_freeze.get('success'):
            md.append(f"- **Full Package List:** [{pip_freeze['package_count']} packages]({Path(pip_freeze['file_path']).name}) ")
        
        md.append("")
        md.append("### Key AI/ML Packages")
        important_packages = ['torch', 'torchvision', 'torchaudio', 'transformers', 'diffusers', 'opencv_python', 'pillow', 'numpy', 'scipy', 'matplotlib']
        md.append("| Package | Installed Version |")
        md.append("|---------|-------------------|")
        for pkg in important_packages:
            if isinstance(python['packages'], dict):
                version = python['packages'].get(pkg, python['packages'].get(pkg.replace('-', '_'), '❌ Not installed'))
            else:
                version = 'Error getting packages'
            md.append(f"| {pkg} | {version} |")
        
        md.append("")
        md.append("### 📦 All Installed Packages")
        md.append("<details>")
        md.append("<summary>Click to expand full package list</summary>")
        md.append("")
        md.append("| Package | Version |")
        md.append("|---------|---------|")
        if isinstance(python['packages'], dict):
            for package, version in sorted(python['packages'].items()):
                md.append(f"| {package} | {version} |")
        md.append("")
        md.append("</details>")
        md.append("")
        
        # ComfyUI-Manager Information
        comfy = data.get('comfyui', {})
        manager_info = comfy.get('manager', {})
        
        md.append("### 🔧 ComfyUI-Manager Information")
        md.append("")
        
        if manager_info.get('exists', False):
            md.append(f"- **Status:** ✅ Installed")
            
            if 'version' in manager_info:
                md.append(f"- **Version:** {manager_info['version']}")
            
            if 'last_update' in manager_info:
                md.append(f"- **Last Update:** {manager_info['last_update']}")
                
            if 'last_commit_message' in manager_info:
                md.append(f"- **Last Commit:** {manager_info['last_commit_message']}")
            
            # Database information
            if 'databases' in manager_info:
                md.append("- **Database Status:**")
                databases = manager_info['databases']
                
                for db_name, db_info in databases.items():
                    if isinstance(db_info, dict) and not db_info.get('error'):
                        entries = db_info.get('entries')
                        size = db_info.get('size_kb', 0)
                        modified = db_info.get('modified', 'Unknown')
                        
                        if entries is not None:
                            md.append(f"  - **{db_name.replace('_', ' ').title()}:** {entries:,} entries ({size} KB, updated {modified})")
                        else:
                            md.append(f"  - **{db_name.replace('_', ' ').title()}:** {size} KB (updated {modified})")
                    else:
                        md.append(f"  - **{db_name.replace('_', ' ').title()}:** ❌ Error or missing")
            
            # Features
            if 'features' in manager_info:
                features = manager_info['features']
                feature_status = []
                if features.get('web_interface'):
                    feature_status.append("Web UI")
                if features.get('cli_tool'):
                    feature_status.append("CLI Tool")
                if features.get('security_check'):
                    feature_status.append("Security Check")
                
                if feature_status:
                    md.append(f"- **Available Features:** {', '.join(feature_status)}")
            
            # Snapshots
            if 'snapshots_count' in manager_info:
                snapshot_count = manager_info['snapshots_count']
                if snapshot_count > 0:
                    md.append(f"- **Snapshots:** {snapshot_count} configuration snapshots available")
                else:
                    md.append(f"- **Snapshots:** No snapshots found")
        else:
            md.append("- **Status:** ❌ ComfyUI-Manager not installed")
            md.append("- **Recommendation:** Install ComfyUI-Manager for easier custom node and model management")
        
        md.append("")
    else:
        md.append("❌ **Python installation not found or error occurred**")
        if 'python' in data and 'error' in data['python']:
            md.append(f"Error: {data['python']['error']}")
    md.append("")
    
    # Custom Nodes Deep Dive
    md.append("## 🔧 Custom Nodes Deep Dive")
    md.append("")
    nodes = data['custom_nodes']
    md.append(f"**Total Custom Nodes:** {len(nodes)}")
    
    # Enhanced statistics
    nodes_with_requirements = sum(1 for node in nodes if node['has_requirements_file'])
    nodes_with_missing = sum(1 for node in nodes if node['missing_packages'])
    nodes_with_conflicts = sum(1 for node in nodes if node['version_conflicts'])
    nodes_with_git = sum(1 for node in nodes if node['git_info'])
    total_size = sum(node['size_mb'] for node in nodes)
    total_python_files = sum(node['python_files'] for node in nodes)
    
    md.append(f"- **Nodes with requirements.txt:** {nodes_with_requirements}")
    md.append(f"- **Nodes with missing dependencies:** {nodes_with_missing}")
    md.append(f"- **Nodes with version conflicts:** {nodes_with_conflicts}")
    md.append(f"- **Nodes with Git repositories:** {nodes_with_git}")
    md.append(f"- **Total disk usage:** {total_size:.1f} MB")
    md.append(f"- **Total Python files:** {total_python_files}")
    md.append("")
    
    # Detailed node analysis table
    md.append("### 📊 Comprehensive Node Analysis")
    md.append("")
    md.append("| Node Name | Size (MB) | Py Files | Git Repo | Last Update | Requirements | Issues |")
    md.append("|-----------|-----------|----------|-----------|-------------|--------------|--------|")
    
    for node in nodes:
        issues = []
        if node['missing_packages']:
            issues.append(f"Missing: {len(node['missing_packages'])}")
        if node['version_conflicts']:
            issues.append(f"Conflicts: {len(node['version_conflicts'])}")
        if not node['has_requirements_file'] and node['python_files'] > 0:
            issues.append("No requirements.txt")
        
        issues_str = ", ".join(issues) if issues else "✅ None"
        req_count = len(node['requirements']) if node['requirements'] else 0
        
        git_status = "✅" if node['git_info'].get('url') else "❌"
        last_update = node['git_info'].get('last_commit_date', 'Unknown')
        if len(last_update) > 10:
            last_update = last_update[:10]
        
        md.append(f"| {node['name']} | {node['size_mb']} | {node['python_files']} | {git_status} | {last_update} | {req_count} | {issues_str} |")
    
    md.append("")
    
    # Dependency Lock Summary
    md.append("## 📦 Dependency Lock Summary")
    md.append("")
    
    if 'dependency_summary' in data:
        dep_summary = data['dependency_summary']
        
        # Missing packages by node
        if dep_summary['missing_by_node']:
            md.append("### ❌ Missing Packages by Node")
            md.append("")
            for node_name, missing_packages in dep_summary['missing_by_node'].items():
                md.append(f"**{node_name}:**")
                for pkg_info in missing_packages:
                    md.append(f"- `{pkg_info['package']}`")
                md.append("")
        
        # Conflicting requirements
        if dep_summary['conflicting_requirements']:
            md.append("### ⚠️ Conflicting Version Requirements")
            md.append("")
            md.append("| Package | Conflicting Requirements | Nodes |")
            md.append("|---------|-------------------------|-------|")
            for package, conflict_info in dep_summary['conflicting_requirements'].items():
                constraints = ", ".join(conflict_info['constraints'])
                nodes_list = ", ".join([req['node'] for req in conflict_info['requirements']])
                md.append(f"| {package} | {constraints} | {nodes_list} |")
            md.append("")
        
        # Most common dependencies
        if dep_summary['most_common']:
            md.append("### 📈 Most Common Dependencies")
            md.append("")
            md.append("| Package | Used by # Nodes |")
            md.append("|---------|-----------------|")
            for package, count in dep_summary['most_common'][:15]:
                md.append(f"| {package} | {count} |")
            md.append("")
    
    # Configuration & Runtime
    md.append("## ⚙️ Configuration & Runtime")
    md.append("")
    
    # Extra model paths configuration FIRST
    if 'config' in comfy:
        config_info = comfy['config']
        if config_info['exists']:
            md.append("### 📁 Extra Model Paths Configuration")
            if 'sections' in config_info:
                md.append(f"- **Configured Sections:** {', '.join(config_info['sections'])}")
            if 'total_paths' in config_info:
                md.append(f"- **Total External Paths:** {config_info['total_paths']}")
            md.append("- ✅ **Status:** Custom model paths configured")
            md.append("")
        else:
            md.append("### 📁 Extra Model Paths Configuration")
            md.append("- ❌ No `extra_model_paths.yaml` found (using default paths only)")
            md.append("")
    
    # Model inventory
    if 'models' in comfy and 'error' not in comfy['models']:
        models_info = comfy['models']
        md.append("### 🎭 Model Inventory")
        md.append("")
        
        # Calculate totals
        total_models = sum(info['count'] for info in models_info.values())
        total_size_gb = sum(info['total_size_gb'] for info in models_info.values())
        
        md.append(f"**Total Models:** {total_models} files ({total_size_gb:.1f} GB)")
        md.append("")
        
        # Separate default and extra paths
        default_models = {k: v for k, v in models_info.items() if k.startswith('default_')}
        extra_models = {k: v for k, v in models_info.items() if not k.startswith('default_')}
        
        # BIG UNIFIED TABLE 1: ALL DEFAULT COMFYUI MODELS
        md.append("#### 📂 Default ComfyUI Models (All Folders)")
        md.append("")
        
        # Collect all default models
        all_default_models = []
        for model_key, model_data in default_models.items():
            if model_data['count'] > 0:
                category = model_key.replace('default_', '').replace('_', ' ').title()
                for model in model_data['models']:
                    model_copy = model.copy()
                    model_copy['category'] = category
                    all_default_models.append(model_copy)
        
        if all_default_models:
            # Sort by size (largest first)
            all_default_models.sort(key=lambda x: x['size_gb'], reverse=True)
            
            default_total_count = len(all_default_models)
            default_total_size = sum(m['size_gb'] for m in all_default_models)
            
            md.append(f"**Total Default Models: {default_total_count} files ({default_total_size:.1f} GB)**")
            md.append("")
            md.append("| Model Name | Category | Size (GB) | Type | Modified | Path |")
            md.append("|------------|----------|-----------|------|----------|------|")
            
            for model in all_default_models:
                path_display = model.get('path', model['name'])[:30] + '...' if len(model.get('path', model['name'])) > 33 else model.get('path', model['name'])
                md.append(f"| {model['name']} | {model['category']} | {model['size_gb']} | {model['extension']} | {model['modified']} | {path_display} |")
            
            md.append("")
        else:
            md.append("*No models found in default ComfyUI directories*")
            md.append("")
        
        # BIG UNIFIED TABLE 2: ALL EXTRA MODELS
        md.append("#### 🔗 Extra Model Paths Models (All External Locations)")
        md.append("")
        
        # Collect all extra models
        all_extra_models = []
        for model_key, model_data in extra_models.items():
            if model_data['count'] > 0:
                # Parse the source and category
                parts = model_key.split('_', 2)
                if len(parts) >= 2:
                    source = parts[0].title()
                    category = '_'.join(parts[1:]).replace('_', ' ').title()
                    display_category = f"{source} - {category}"
                else:
                    display_category = model_key.replace('_', ' ').title()
                
                for model in model_data['models']:
                    model_copy = model.copy()
                    model_copy['category'] = display_category
                    all_extra_models.append(model_copy)
        
        if all_extra_models:
            # Sort by size (largest first)
            all_extra_models.sort(key=lambda x: x['size_gb'], reverse=True)
            
            extra_total_count = len(all_extra_models)
            extra_total_size = sum(m['size_gb'] for m in all_extra_models)
            
            md.append(f"**Total Extra Models: {extra_total_count} files ({extra_total_size:.1f} GB)**")
            md.append("")
            md.append("| Model Name | Source & Category | Size (GB) | Type | Modified | Path |")
            md.append("|------------|-------------------|-----------|------|----------|------|")
            
            for model in all_extra_models:
                path_display = model.get('path', model['name'])[:30] + '...' if len(model.get('path', model['name'])) > 33 else model.get('path', model['name'])
                md.append(f"| {model['name']} | {model['category']} | {model['size_gb']} | {model['extension']} | {model['modified']} | {path_display} |")
            
            md.append("")
        else:
            md.append("*No models found in extra model paths*")
            md.append("")
        
        # PRIORITY MODEL TYPES SUMMARY - Simple counts
        priority_types = ['checkpoints', 'diffusion_models', 'loras', 'unet', 'text_encoders']
        
        md.append("#### 🎯 Priority Model Types Summary")
        md.append("")
        md.append("| Model Type | Default Count | Extra Count | Total Files | Total Size (GB) |")
        md.append("|------------|---------------|-------------|-------------|-----------------|")
        
        for priority_type in priority_types:
            # Count default models
            default_key = f"default_{priority_type}"
            default_count = models_info.get(default_key, {}).get('count', 0)
            default_size = models_info.get(default_key, {}).get('total_size_gb', 0.0)
            
            # Count extra models
            extra_count = 0
            extra_size = 0.0
            for key, info in extra_models.items():
                if priority_type in key.lower():
                    extra_count += info['count']
                    extra_size += info['total_size_gb']
            
            total_count = default_count + extra_count
            total_size = default_size + extra_size
            
            status_icon = "✅" if total_count > 0 else "❌"
            display_name = priority_type.replace('_', ' ').title()
            
            md.append(f"| {status_icon} **{display_name}** | {default_count} | {extra_count} | {total_count} | {total_size:.1f} |")
        
        md.append("")
        
        # Complete summary for all model types
        md.append("#### 📊 Complete Model Type Summary")
        md.append("")
        md.append("| Model Type | Default Count | Extra Count | Total Files | Total Size (GB) |")
        md.append("|------------|---------------|-------------|-------------|-----------------|")
        
        # Get all unique model types
        all_types = set()
        for key in models_info.keys():
            if key.startswith('default_'):
                all_types.add(key.replace('default_', ''))
            else:
                parts = key.split('_', 1)
                if len(parts) >= 2:
                    all_types.add(parts[1])
        
        for model_type in sorted(all_types):
            default_key = f"default_{model_type}"
            default_info = models_info.get(default_key, {'count': 0, 'total_size_gb': 0.0})
            
            # Sum up all extra model instances for this type
            extra_count = 0
            extra_size = 0.0
            for key, info in extra_models.items():
                if model_type in key.lower():
                    extra_count += info['count']
                    extra_size += info['total_size_gb']
            
            total_count = default_info['count'] + extra_count
            total_size = default_info['total_size_gb'] + extra_size
            
            if total_count > 0:  # Only show types that have models
                md.append(f"| **{model_type.replace('_', ' ').title()}** | {default_info['count']} | {extra_count} | {total_count} | {total_size:.1f} |")
        
        md.append("")
    
    # Log analysis
    if 'logs' in comfy:
        log_info = comfy['logs']
        if log_info['found_logs']:
            md.append("### 📄 Log Files")
            md.append("| File | Size (KB) | Last Modified |")
            md.append("|------|-----------|---------------|")
            for log in log_info['found_logs'][:5]:  # Show first 5 logs
                size = log.get('size_kb', 'Unknown')
                modified = log.get('modified', 'Unknown')
                md.append(f"| {Path(log['path']).name} | {size} | {modified} |")
            md.append("")
            
            if log_info['errors_warnings']:
                md.append("### ⚠️ Recent Errors/Warnings from Logs")
                md.append("```")
                for entry in log_info['errors_warnings'][:10]:  # Show first 10
                    md.append(f"[{entry['file']}:{entry.get('line_number', '?')}] {entry['line'][:100]}...")
                md.append("```")
                md.append("")
    
    # Recommendations
    md.append("## 💡 Recommendations")
    md.append("")
    
    recommendations = []
    
    if nodes_with_missing > 0:
        recommendations.append("🔧 **Install missing dependencies** - Run `pip install` for missing packages")
    
    if nodes_with_conflicts > 0:
        recommendations.append("⚠️ **Resolve version conflicts** - Update or downgrade conflicting packages")
    
    if not comfy['exists']:
        recommendations.append("❌ **ComfyUI directory not found** - Verify installation path")
    
    python_issues = 'python' not in data or 'error' in data.get('python', {})
    if python_issues:
        recommendations.append("🐍 **Python environment issues** - Check Python installation and PATH")
    
    large_nodes = [n for n in nodes if n['size_mb'] > 100]
    if large_nodes:
        recommendations.append(f"💾 **Large nodes detected** - Consider cleaning up {len(large_nodes)} nodes > 100MB")
    
    outdated_nodes = [n for n in nodes if n['git_info'].get('last_commit_date') and n['git_info']['last_commit_date'] < '2024-01-01']
    if outdated_nodes:
        recommendations.append(f"📅 **Outdated nodes** - {len(outdated_nodes)} nodes haven't been updated in over a year")
    
    if 'gpu_info' in python_info and not gpu_info.get('pytorch_cuda_info'):
        recommendations.append("🎮 **GPU support** - PyTorch CUDA not detected, consider installing GPU-enabled PyTorch")
    
    if not recommendations:
        recommendations.append("✅ **System looks excellent!** No major issues detected.")
    
    for rec in recommendations:
        md.append(f"- {rec}")
    
    md.append("")
    md.append("---")
    md.append("")
    md.append(f"*🤖 Report generated by ComfyUI-Ultimate-Inspector*")
    md.append(f"*📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 📁 Output directory: {output_dir}*")
    
    return "\n".join(md)

def main():
    parser = argparse.ArgumentParser(description='ComfyUI Ultimate System Inspector')
    parser.add_argument('--path', '-p', default='.', help='Root path to ComfyUI installation (default: current directory)')
    parser.add_argument('--output', '-o', default='comfyui_ultimate_report.md', help='Output file path (default: comfyui_ultimate_report.md)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--output-dir', default='comfyui_analysis', help='Directory for additional output files (default: comfyui_analysis)')
    
    args = parser.parse_args()
    
    # Use the same directory as the markdown report for all outputs
    output_file = Path(args.output)
    output_dir = output_file.parent
    output_dir.mkdir(exist_ok=True)
    
    # Get root path
    root_path = Path(args.path).resolve()
    
    if args.verbose:
        print(f"[*] Starting ultimate ComfyUI analysis at: {root_path}")
        print(f"[*] Output directory: {output_dir}")
    
    # Collect basic data
    data = {
        'timestamp': datetime.now().isoformat(),
        'root_path': str(root_path),
        'system': get_system_info_enhanced(),
    }
    
    if args.verbose:
        print("[*] Analyzing ComfyUI installation...")
    data['comfyui'] = get_comfyui_info_enhanced(root_path)
    
    # Find and analyze Python
    python_path = find_python_executable(root_path)
    if python_path:
        if args.verbose:
            print(f"[*] Analyzing Python environment: {python_path}")
        data['python'] = get_python_info_enhanced(python_path, output_dir)
        data['python']['path'] = str(python_path)
        
        # Analyze custom nodes with enhanced dependency checking
        if args.verbose:
            print("[*] Deep-diving into custom nodes...")
        installed_packages = data['python'].get('packages', {})
        data['custom_nodes'] = get_custom_nodes_enhanced(root_path, installed_packages)
        
        # Generate dependency summary
        if args.verbose:
            print("[*] Analyzing dependencies...")
        data['dependency_summary'] = generate_dependency_lock_summary(data['custom_nodes'])
    else:
        data['python'] = {'error': 'No Python executable found'}
        data['custom_nodes'] = []
        data['dependency_summary'] = {}
        if args.verbose:
            print("[!] No Python executable found")
    
    # Generate report
    report = generate_ultimate_markdown_report(data, output_dir)
    
    # Save main report
    output_file = Path(args.output)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"[+] Ultimate analysis report saved to: {output_file}")
    
    # Print executive summary
    nodes = data['custom_nodes']
    nodes_with_issues = sum(1 for node in nodes if node['missing_packages'] or node['version_conflicts'])
    
    print(f"[+] Executive Summary:")
    print(f"    {len(nodes)} custom nodes analyzed")
    print(f"    {nodes_with_issues} nodes with dependency issues")
    
    # Confirm pip_freeze.txt location
    if 'pip_freeze' in data.get('python', {}) and data['python']['pip_freeze'].get('success'):
        freeze_file_path = data['python']['pip_freeze']['file_path']
        package_count = data['python']['pip_freeze']['package_count']
        print(f"[+] Requirements file saved to: {freeze_file_path} ({package_count} packages)")
    
    print(f"[+] All output files saved to directory: {output_dir}")

if __name__ == "__main__":
    main()
