# ---
# deploy: true
# cmd: ["modal", "serve", "modal_reactor_app.py"]
# ---

# ReActor Face Video Swap Modal App
# This app runs ComfyUI with ReActor and VHS nodes for face swapping in videos

import json
import subprocess
import uuid
import os
from pathlib import Path
from typing import Dict, Optional, Annotated

import modal
import modal.experimental
import fastapi


# Only need the ReActor models volume
reactor_models_vol = modal.Volume.from_name("Reactor-comfy-files", create_if_missing=True)

# Build the Modal image with minimal dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git", 
        "ffmpeg",  # Required for video processing
        "libgl1-mesa-glx",  # OpenCV dependencies
        "libglib2.0-0"
    )
    .pip_install(
        "fastapi[standard]==0.115.4",
        "uvicorn[standard]==0.32.1",
        "filetype",
        "Pillow",
        "opencv-python-headless",
        "numpy",
        "comfy-cli==1.4.1"
    )
    .run_commands(
        # Install ComfyUI (using latest version)
        "comfy --skip-prompt install --fast-deps --nvidia",
        # Remove any existing installations to avoid conflicts
        "rm -rf /root/comfy/ComfyUI/custom_nodes/ComfyUI-ReActor",
        "rm -rf /root/comfy/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite"
    )
    .run_commands(
        # Clone ReActor directly from GitHub
        "cd /root/comfy/ComfyUI/custom_nodes && git clone https://github.com/Gourieff/ComfyUI-ReActor.git",
        # Clone VHS directly from GitHub  
        "cd /root/comfy/ComfyUI/custom_nodes && git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"
    )
    .run_commands(
        # Install ReActor dependencies using their requirements.txt
        "cd /root/comfy/ComfyUI/custom_nodes/ComfyUI-ReActor && pip install -r requirements.txt",
        # Install VHS dependencies using their requirements.txt
        "cd /root/comfy/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite && pip install -r requirements.txt",
        # Install additional dependencies that might be missing
        "pip install insightface onnxruntime-gpu torch torchvision torchaudio"
    )
    .run_commands(
        # Install additional dependencies that are commonly missing
        "pip install opencv-python-headless numpy scipy scikit-image",
        "pip install ultralytics segment-anything",
        "pip install gfpgan basicsr facexlib",
        "pip install codeformer",
        # Ensure all dependencies are properly installed
        "pip install --upgrade pip setuptools wheel"
    )
)

# Function to set up model symlinks
def setup_reactor_models():
    """Set up symlinks for ReActor models in ComfyUI directories"""
    comfyui_base = "/root/comfy/ComfyUI"
    
    # Create necessary directories
    os.makedirs(f"{comfyui_base}/models/insightface", exist_ok=True)
    os.makedirs(f"{comfyui_base}/models/ultralytics/bbox", exist_ok=True)
    os.makedirs(f"{comfyui_base}/models/sams", exist_ok=True)
    os.makedirs(f"{comfyui_base}/models/facerestore_models", exist_ok=True)
    os.makedirs(f"{comfyui_base}/models/detection", exist_ok=True)
    
    # Symlink models from volume to ComfyUI directories
    # Updated paths to match the actual volume structure
    model_mappings = [
        ("/models/insightface/inswapper_128.onnx", f"{comfyui_base}/models/insightface/inswapper_128.onnx"),
        ("/models/insightface/inswapper_128_fp16.onnx", f"{comfyui_base}/models/insightface/inswapper_128_fp16.onnx"),
        ("/models/ultralytics/bbox/face_yolov8m.pt", f"{comfyui_base}/models/ultralytics/bbox/face_yolov8m.pt"),
        ("/models/sams/sam_vit_l_0b3195.pth", f"{comfyui_base}/models/sams/sam_vit_l_0b3195.pth"),
        ("/models/facerestore_models/GFPGANv1.4.pth", f"{comfyui_base}/models/facerestore_models/GFPGANv1.4.pth"),
        ("/models/facerestore_models/codeformer-v0.1.0.pth", f"{comfyui_base}/models/facerestore_models/codeformer-v0.1.0.pth"),
        # The detection directory structure is different - it has a .cache subdirectory
        # We need to find the buffalo_l.zip file in the detection directory
    ]
    
    for src, dst in model_mappings:
        if os.path.exists(src):
            if os.path.exists(dst):
                os.remove(dst)  # Remove existing symlink/file
            os.symlink(src, dst)
            print(f"✅ Linked {os.path.basename(src)} to ComfyUI")
        else:
            print(f"⚠️ Source file not found: {src}")
    
    # Handle detection models separately since they have a different structure
    detection_dir = "/models/detection"
    if os.path.exists(detection_dir):
        print(f"🔍 Checking detection directory: {detection_dir}")
        
        # First, let's see what's actually in the detection directory
        print("📁 Contents of detection directory:")
        for item in os.listdir(detection_dir):
            item_path = os.path.join(detection_dir, item)
            if os.path.isfile(item_path):
                print(f"  📄 {item} ({os.path.getsize(item_path)} bytes)")
            elif os.path.isdir(item_path):
                print(f"  📁 {item}/")
        
        # Look for detection models in the models subdirectory
        detection_models_dir = os.path.join(detection_dir, "models")
        if os.path.exists(detection_models_dir):
            print(f"🔍 Checking detection models directory: {detection_models_dir}")
            detection_models = []
            for item in os.listdir(detection_models_dir):
                item_path = os.path.join(detection_models_dir, item)
                if os.path.isfile(item_path) and (item.endswith('.zip') or item.endswith('.pth') or item.endswith('.onnx')):
                    detection_models.append((item, item_path))
            
            if detection_models:
                print(f"✅ Found {len(detection_models)} detection model(s)")
                for model_name, model_path in detection_models:
                    dst_path = f"{comfyui_base}/models/detection/{model_name}"
                    if os.path.exists(dst_path):
                        os.remove(dst_path)
                    os.symlink(model_path, dst_path)
                    print(f"✅ Linked detection model {model_name} to ComfyUI")
            else:
                print("⚠️ No detection model files found in detection/models directory")
        else:
            print("⚠️ Detection models directory not found")
            
        # Also check the .cache subdirectory
        cache_dir = os.path.join(detection_dir, ".cache")
        if os.path.exists(cache_dir):
            print(f"🔍 Checking detection cache directory: {cache_dir}")
            cache_models = []
            for item in os.listdir(cache_dir):
                item_path = os.path.join(cache_dir, item)
                if os.path.isfile(item_path) and (item.endswith('.zip') or item.endswith('.pth') or item.endswith('.onnx')):
                    cache_models.append((item, item_path))
            
            if cache_models:
                print(f"✅ Found {len(cache_models)} detection model(s) in cache")
                for model_name, model_path in cache_models:
                    dst_path = f"{comfyui_base}/models/detection/{model_name}"
                    if os.path.exists(dst_path):
                        os.remove(dst_path)
                    os.symlink(model_path, dst_path)
                    print(f"✅ Linked detection model {model_name} to ComfyUI (from cache)")
    else:
        print("❌ Detection directory not found")

# Function to verify ComfyUI installation
def verify_comfyui_installation():
    """Verify that ComfyUI and custom nodes are properly installed"""
    comfy_dir = "/root/comfy/ComfyUI"
    
    # Check if ComfyUI is installed
    if not os.path.exists(comfy_dir):
        print("❌ ComfyUI directory not found")
        return False
    
    # Check if custom nodes are installed
    custom_nodes_dir = f"{comfy_dir}/custom_nodes"
    if not os.path.exists(custom_nodes_dir):
        print("❌ Custom nodes directory not found")
        return False
    
    # Check for ReActor
    reactor_dir = f"{custom_nodes_dir}/ComfyUI-ReActor"
    if not os.path.exists(reactor_dir):
        print("❌ ReActor not found")
        return False
    
    # Check for VHS
    vhs_dir = f"{custom_nodes_dir}/ComfyUI-VideoHelperSuite"
    if not os.path.exists(vhs_dir):
        print("❌ VHS not found")
        return False
    
    # Test basic ComfyUI functionality
    try:
        result = subprocess.run(
            "comfy --help", 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        if result.returncode != 0:
            print("❌ ComfyUI command failed")
            return False
        print("✅ ComfyUI installation verified")
        return True
    except Exception as e:
        print(f"❌ Error testing ComfyUI: {str(e)}")
        return False

# Add the setup function to the image
image = image.run_function(
    setup_reactor_models,
    volumes={"/models": reactor_models_vol}
)

# Add the verification function to the image
image = image.run_function(
    verify_comfyui_installation,
    volumes={"/models": reactor_models_vol}
)

# Add the workflow file to the container
image = image.add_local_file(
    Path(__file__).parent / "face-video-swapv2.json", "/root/face-video-swapv2.json"
)

# Create the Modal app
app = modal.App(name="reactor-face-swap", image=image)

# Interactive UI function for development
@app.function(
    max_containers=1,
    gpu="L40S",
    volumes={"/models": reactor_models_vol},
    timeout=3600
)
@modal.web_server(8000, startup_timeout=120)
def ui():
    """Interactive ComfyUI interface for development"""
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)

# Main API class for face video swapping
@app.cls(
    scaledown_window=300,
    gpu="L40S",
    volumes={"/models": reactor_models_vol},
    timeout=3600
)
class ReActorFaceSwap:
    port: int = 8000

    @modal.enter()
    def launch_comfy_background(self):
        """Launch ComfyUI server in background"""
        cmd = f"comfy launch --background -- --port {self.port}"
        subprocess.run(cmd, shell=True, check=True)
        print("✅ ComfyUI server launched")

    @modal.method()
    def infer(self, workflow_path: str = "/root/face-video-swapv2.json"):
        """Run inference on a workflow"""
        print("🚀 Starting inference process...")
        print("🔍 DEBUG: Enhanced logging is active!")
        print(f"🔍 DEBUG: Workflow path: {workflow_path}")
        print(f"🔍 DEBUG: Current time: {__import__('datetime').datetime.now()}")
        
        # Check server health
        print("🔍 Checking ComfyUI server health...")
        self.poll_server_health()
        
        print(f"📁 Executing workflow: {workflow_path}")
        
        # Check if workflow file exists
        print(f"🔍 Checking workflow file existence: {workflow_path}")
        if not os.path.exists(workflow_path):
            print(f"❌ Workflow file not found: {workflow_path}")
            raise Exception(f"Workflow file not found: {workflow_path}")
        else:
            print(f"✅ Workflow file exists: {workflow_path}")
            # Check file size
            file_size = os.path.getsize(workflow_path)
            print(f"📊 Workflow file size: {file_size} bytes")
        
        # List available models for debugging
        print("🔍 Checking available models:")
        models_dir = "/root/comfy/ComfyUI/models"
        if os.path.exists(models_dir):
            print(f"✅ Models directory exists: {models_dir}")
            for root, dirs, files in os.walk(models_dir):
                level = root.replace(models_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    print(f"{subindent}{file} ({file_size} bytes)")
        else:
            print("❌ Models directory not found")
        
        # Read and validate workflow
        print("📋 Loading and validating workflow...")
        try:
            with open(workflow_path, 'r') as f:
                workflow_data = json.load(f)
            print(f"✅ Workflow loaded successfully with {len(workflow_data)} nodes")
            
            # Print the workflow JSON for debugging
            print("📋 Workflow JSON:")
            print(json.dumps(workflow_data, indent=2))
            
            # Verify specific model paths mentioned in workflow
            print("🔍 Verifying model paths in workflow...")
            if "4" in workflow_data and "inputs" in workflow_data["4"]:
                inputs = workflow_data["4"]["inputs"]
                
                # Check swap model
                swap_model = inputs.get("swap_model", "")
                swap_model_path = f"/root/comfy/ComfyUI/models/{swap_model}"
                print(f"🔍 Checking swap model: {swap_model}")
                if os.path.exists(swap_model_path):
                    print(f"✅ Swap model exists: {swap_model_path}")
                    print(f"📊 Swap model size: {os.path.getsize(swap_model_path)} bytes")
                else:
                    print(f"❌ Swap model not found: {swap_model_path}")
                
                # Check face detection model
                face_detection = inputs.get("facedetection", "")
                face_detection_path = f"/root/comfy/ComfyUI/models/{face_detection}"
                print(f"🔍 Checking face detection model: {face_detection}")
                if os.path.exists(face_detection_path):
                    print(f"✅ Face detection model exists: {face_detection_path}")
                    print(f"📊 Face detection model size: {os.path.getsize(face_detection_path)} bytes")
                else:
                    print(f"❌ Face detection model not found: {face_detection_path}")
                
                # Check face restore model
                face_restore = inputs.get("face_restore_model", "")
                if face_restore != "none":
                    face_restore_path = f"/root/comfy/ComfyUI/models/{face_restore}"
                    print(f"🔍 Checking face restore model: {face_restore}")
                    if os.path.exists(face_restore_path):
                        print(f"✅ Face restore model exists: {face_restore_path}")
                        print(f"📊 Face restore model size: {os.path.getsize(face_restore_path)} bytes")
                    else:
                        print(f"❌ Face restore model not found: {face_restore_path}")
                else:
                    print("ℹ️ Face restore model set to 'none'")
            
        except Exception as e:
            print(f"❌ Failed to load workflow: {str(e)}")
            raise Exception(f"Failed to load workflow: {str(e)}")
        
        # Check ComfyUI environment
        print("🔍 Checking ComfyUI environment...")
        comfy_dir = "/root/comfy/ComfyUI"
        if os.path.exists(comfy_dir):
            print(f"✅ ComfyUI directory exists: {comfy_dir}")
            
            # Check Python path
            print(f"🔍 Python path: {os.environ.get('PYTHONPATH', 'Not set')}")
            
            # Check if we're in the right directory
            current_dir = os.getcwd()
            print(f"🔍 Current working directory: {current_dir}")
            
            # Check ComfyUI version/info
            try:
                import sys
                sys.path.insert(0, comfy_dir)
                print("✅ Added ComfyUI to Python path")
            except Exception as e:
                print(f"⚠️ Could not add ComfyUI to Python path: {str(e)}")
        else:
            print(f"❌ ComfyUI directory not found: {comfy_dir}")
        
        # Run the workflow
        print("🚀 Starting workflow execution...")
        try:
            # First, let's check if the custom nodes are loaded
            print("🔍 Checking custom nodes:")
            custom_nodes_dir = "/root/comfy/ComfyUI/custom_nodes"
            if os.path.exists(custom_nodes_dir):
                print(f"✅ Custom nodes directory exists: {custom_nodes_dir}")
                for item in os.listdir(custom_nodes_dir):
                    item_path = os.path.join(custom_nodes_dir, item)
                    if os.path.isdir(item_path):
                        print(f"  📁 {item}/")
                        # Check for Python files
                        py_files = [f for f in os.listdir(item_path) if f.endswith('.py')]
                        for py_file in py_files[:3]:  # Show first 3 files
                            print(f"    📄 {py_file}")
                        if len(py_files) > 3:
                            print(f"    ... and {len(py_files) - 3} more files")
                        
                        # Check for __init__.py files
                        if "__init__.py" in py_files:
                            print(f"    ✅ {item} has __init__.py")
                        else:
                            print(f"    ⚠️ {item} missing __init__.py")
            else:
                print("❌ Custom nodes directory not found")
            
            # Test importing the custom nodes to see if there are any import errors
            print("🔍 Testing custom node imports...")
            try:
                import sys
                sys.path.insert(0, comfy_dir)
                sys.path.insert(0, custom_nodes_dir)
                
                # Test ReActor import
                try:
                    from ComfyUI_ReActor import nodes
                    print("✅ ReActor nodes imported successfully")
                except Exception as e:
                    print(f"❌ Failed to import ReActor nodes: {str(e)}")
                
                # Test VHS import
                try:
                    from ComfyUI_VideoHelperSuite import nodes
                    print("✅ VHS nodes imported successfully")
                except Exception as e:
                    print(f"❌ Failed to import VHS nodes: {str(e)}")
                
            except Exception as e:
                print(f"❌ Error testing imports: {str(e)}")
            
            # Check if ComfyUI can load the nodes
            print("🔍 Testing ComfyUI node loading...")
            try:
                # Try to run a simple ComfyUI command to test node loading
                test_cmd = "comfy --help"
                test_result = subprocess.run(test_cmd, shell=True, capture_output=True, text=True, timeout=30)
                print(f"✅ ComfyUI command test successful: {test_result.returncode}")
                
                # Try to list available nodes
                list_cmd = "comfy node list"
                list_result = subprocess.run(list_cmd, shell=True, capture_output=True, text=True, timeout=30)
                print(f"📋 Available nodes: {len(list_result.stdout.splitlines())} lines")
                if "ReActor" in list_result.stdout:
                    print("✅ ReActor nodes found in ComfyUI")
                else:
                    print("❌ ReActor nodes not found in ComfyUI")
                if "VHS" in list_result.stdout:
                    print("✅ VHS nodes found in ComfyUI")
                else:
                    print("❌ VHS nodes not found in ComfyUI")
                    
            except Exception as e:
                print(f"❌ Error testing ComfyUI: {str(e)}")
            
            # Check ComfyUI logs directory for any error logs
            print("🔍 Checking ComfyUI logs:")
            logs_dir = "/root/comfy/ComfyUI/logs"
            if os.path.exists(logs_dir):
                print(f"✅ Logs directory exists: {logs_dir}")
                log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
                for log_file in sorted(log_files, reverse=True)[:3]:  # Show last 3 log files
                    log_path = os.path.join(logs_dir, log_file)
                    print(f"  📄 {log_file}")
                    try:
                        with open(log_path, 'r') as f:
                            last_lines = f.readlines()[-10:]  # Last 10 lines
                            for line in last_lines:
                                if 'error' in line.lower() or 'exception' in line.lower():
                                    print(f"    ⚠️ {line.strip()}")
                    except Exception as e:
                        print(f"    ❌ Could not read log: {str(e)}")
            else:
                print("❌ Logs directory not found")
            
            # Set up environment for ComfyUI
            env = dict(os.environ)
            env['PYTHONPATH'] = f"{comfy_dir}:{env.get('PYTHONPATH', '')}"
            print(f"🔍 Set PYTHONPATH: {env['PYTHONPATH']}")
            
            # Test with a simple workflow first
            print("🔍 Testing with a simple workflow...")
            test_workflow = {
                "1": {
                    "inputs": {
                        "text": "test"
                    },
                    "class_type": "CLIPTextEncode"
                }
            }
            test_workflow_path = "/tmp/test_workflow.json"
            with open(test_workflow_path, 'w') as f:
                json.dump(test_workflow, f)
            
            test_cmd = f"comfy run --workflow {test_workflow_path} --wait --timeout 60 --verbose"
            print(f"🚀 Running test command: {test_cmd}")
            
            test_result = subprocess.run(
                test_cmd, 
                shell=True, 
                capture_output=True, 
                text=True,
                timeout=60,
                env=env,
                cwd=comfy_dir
            )
            
            print(f"📊 Test command completed with return code: {test_result.returncode}")
            if test_result.returncode != 0:
                print(f"❌ Test command failed:")
                print(f"STDOUT: {test_result.stdout}")
                print(f"STDERR: {test_result.stderr}")
                raise Exception("Basic ComfyUI test failed - there's a fundamental issue with the installation")
            else:
                print("✅ Basic ComfyUI test passed")
            
            cmd = f"comfy run --workflow {workflow_path} --wait --timeout 1800 --verbose"
            print(f"🚀 Running command: {cmd}")
            
            # Run with more detailed error capture
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True,
                timeout=1800,
                env=env,
                cwd=comfy_dir
            )
            
            print(f"📊 Command completed with return code: {result.returncode}")
            print(f"📊 STDOUT length: {len(result.stdout)} characters")
            print(f"📊 STDERR length: {len(result.stderr)} characters")
            
            if result.returncode != 0:
                print(f"❌ Command failed with return code: {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                
                # Try to get more detailed error info
                if "An unknown error occurred" in result.stdout:
                    print("🔍 Attempting to get more detailed error info...")
                    # Check if there are any Python tracebacks in the output
                    lines = result.stdout.split('\n')
                    for i, line in enumerate(lines):
                        if 'Traceback' in line or 'Error:' in line or 'Exception:' in line:
                            print(f"Found error at line {i}: {line}")
                            # Show next few lines for context
                            for j in range(i, min(i+10, len(lines))):
                                print(f"  {lines[j]}")
                
                # Check for specific error patterns
                if "ModuleNotFoundError" in result.stderr:
                    print("🔍 ModuleNotFoundError detected - missing dependency")
                if "ImportError" in result.stderr:
                    print("🔍 ImportError detected - import issue")
                if "FileNotFoundError" in result.stderr:
                    print("🔍 FileNotFoundError detected - missing file")
                if "PermissionError" in result.stderr:
                    print("🔍 PermissionError detected - permission issue")
                
                raise Exception(f"Command failed: {result.stderr}")
            
            print("✅ Workflow executed successfully")
            
        except subprocess.TimeoutExpired:
            print("⏰ Workflow execution timed out after 30 minutes")
            raise Exception("Workflow execution timed out after 30 minutes")
        except Exception as e:
            print(f"❌ Error running workflow: {str(e)}")
            raise Exception(f"Error running workflow: {str(e)}")
        
        # Collect outputs
        print("📦 Collecting outputs...")
        outputs = []
        output_dir = "/root/comfy/ComfyUI/output"
        temp_dir = "/root/comfy/ComfyUI/temp"
        
        print(f"🔍 Checking output directory: {output_dir}")
        print(f"🔍 Checking temp directory: {temp_dir}")
        
        # Check output directory for images
        if os.path.exists(output_dir):
            files = list(Path(output_dir).iterdir())
            print(f"Found {len(files)} files in output directory")
            for f in files:
                if f.is_file():
                    print(f"  - {f.name} ({f.stat().st_size} bytes)")
                    outputs.append({
                        "type": "output",
                        "name": f.name,
                        "data": f.read_bytes()
                    })
        else:
            print("❌ Output directory not found")
        
        # Check temp directory for videos
        if os.path.exists(temp_dir):
            files = list(Path(temp_dir).iterdir())
            print(f"Found {len(files)} files in temp directory")
            for f in files:
                if f.is_file() and f.suffix in ['.mp4', '.avi', '.mov']:
                    print(f"  - {f.name} ({f.stat().st_size} bytes)")
                    outputs.append({
                        "type": "video",
                        "name": f.name,
                        "data": f.read_bytes()
                    })
        else:
            print("❌ Temp directory not found")
        
        if not outputs:
            print("⚠️ No outputs found")
            # List all files in both directories for debugging
            if os.path.exists(output_dir):
                print("Output directory contents:")
                for f in Path(output_dir).iterdir():
                    print(f"  - {f.name}")
            if os.path.exists(temp_dir):
                print("Temp directory contents:")
                for f in Path(temp_dir).iterdir():
                    print(f"  - {f.name}")
        else:
            print(f"✅ Found {len(outputs)} outputs")
        
        print("🏁 Inference process completed")
        return outputs

    @modal.fastapi_endpoint(method="POST")
    def api(
        self,
        source_image: Annotated[bytes, fastapi.File(description="Source face image")],
        target_video: Annotated[bytes, fastapi.File(description="Target video for face swap")]
    ):
        """API endpoint for face video swapping with file uploads"""
        from fastapi import Response, HTTPException
        import tempfile
        import shutil
        import filetype
        from PIL import Image
        import io
        import cv2
        import os

        try:
            # Comprehensive file validation (inspired by FaceFusion)
            def validate_image_type(image_bytes: bytes, label: str):
                kind = filetype.guess(image_bytes)
                if kind is None or not kind.mime.startswith("image/"):
                    raise HTTPException(status_code=400, detail=f"[{label}] Uploaded file is not a valid image.")
                if kind.mime == "image/gif":
                    raise HTTPException(status_code=400, detail=f"[{label}] GIF format is not supported for face swapping.")

            def validate_video_type(video_bytes: bytes, label: str):
                kind = filetype.guess(video_bytes)
                if kind is None or not kind.mime.startswith("video/"):
                    raise HTTPException(status_code=400, detail=f"[{label}] Uploaded file is not a valid video.")
                if kind.mime == "video/gif":
                    raise HTTPException(status_code=400, detail=f"[{label}] GIF format is not supported for face swapping.")

            def validate_image_openable(image_bytes: bytes, label: str):
                try:
                    Image.open(io.BytesIO(image_bytes)).verify()
                except Exception:
                    raise HTTPException(status_code=400, detail=f"[{label}] Uploaded file is not readable as an image.")

            def validate_video_openable(video_bytes: bytes, label: str):
                try:
                    # Save to temporary file first
                    temp_path = 'temp_validate.mp4'
                    with open(temp_path, 'wb') as f:
                        f.write(video_bytes)
                    
                    # Try to open with OpenCV
                    cap = cv2.VideoCapture(temp_path)
                    if not cap.isOpened():
                        raise Exception("Video could not be opened with OpenCV")
                    
                    # Try to read first frame
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        raise Exception("Could not read first frame")
                    
                    # Get video properties
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    if fps <= 0 or frame_count <= 0:
                        raise Exception("Invalid video properties")
                    
                    cap.release()
                    os.remove(temp_path)
                    
                except Exception as e:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    raise HTTPException(status_code=400, detail=f"[{label}] Uploaded file is not readable as a video: {str(e)}")

            # Validate inputs
            validate_image_type(source_image, "source")
            validate_image_openable(source_image, "source")
            validate_video_type(target_video, "target")
            validate_video_openable(target_video, "target")

            # Create temporary directory for uploaded files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)
                
                # Save uploaded files with proper extensions
                source_image_path = temp_dir_path / "source_face.jpg"
                target_video_path = temp_dir_path / "target_video.mp4"
                
                # Write uploaded files to temp directory
                with open(source_image_path, "wb") as f:
                    f.write(source_image)
                
                with open(target_video_path, "wb") as f:
                    f.write(target_video)
                
                # Check if workflow file exists
                workflow_file = Path(__file__).parent / "face-video-swapv2.json"
                if not workflow_file.exists():
                    # Try alternative locations
                    alt_locations = [
                        "/root/face-video-swapv2.json",
                        Path.cwd() / "face-video-swapv2.json",
                        Path.home() / "face-video-swapv2.json"
                    ]
                    
                    workflow_file = None
                    for loc in alt_locations:
                        if Path(loc).exists():
                            workflow_file = Path(loc)
                            break
                    
                    if workflow_file is None:
                        raise HTTPException(
                            status_code=500, 
                            detail="Workflow file 'face-video-swapv2.json' not found in any expected location"
                        )
                
                # Load the workflow
                try:
                    workflow_data = json.loads(workflow_file.read_text())
                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to load workflow file: {str(e)}"
                    )

                # Update workflow with uploaded files
                client_id = uuid.uuid4().hex
                
                # Update source image (node 5 - LoadImage)
                workflow_data["5"]["inputs"]["image"] = str(source_image_path)
                
                # Update target video (node 1 - VHS_LoadVideoPath)
                workflow_data["1"]["inputs"]["video"] = str(target_video_path)
                
                # Update output filename prefix (node 6 - VHS_VideoCombine)
                workflow_data["6"]["inputs"]["filename_prefix"] = f"faceswap_{client_id}"

                # Save updated workflow
                new_workflow_file = temp_dir_path / f"{client_id}.json"
                with open(new_workflow_file, "w") as f:
                    json.dump(workflow_data, f, indent=2)

                # Run inference
                outputs = self.infer.local(str(new_workflow_file))

                # Return results
                if outputs:
                    # Look for video output first
                    for output in outputs:
                        if output["type"] == "video":
                            return Response(
                                content=output["data"], 
                                media_type="video/mp4",
                                headers={
                                    "Content-Disposition": f"attachment; filename={output['name']}",
                                    "X-Client-ID": client_id
                                }
                            )
                    
                    # If no video, return first output
                    return Response(
                        content=outputs[0]["data"],
                        media_type="application/octet-stream",
                        headers={
                            "Content-Disposition": f"attachment; filename={outputs[0]['name']}",
                            "X-Client-ID": client_id
                        }
                    )
                else:
                    raise HTTPException(status_code=500, detail="No output generated")

        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    @modal.fastapi_endpoint(method="GET")
    def health(self):
        """Health check endpoint"""
        return {"status": "healthy", "message": "ReActor Face Swap API is running"}

    def poll_server_health(self) -> Dict:
        """Check if ComfyUI server is healthy"""
        import socket
        import urllib

        try:
            req = urllib.request.Request(f"http://127.0.0.1:{self.port}/system_stats")
            urllib.request.urlopen(req, timeout=10)
            print("✅ ComfyUI server is healthy")
        except (socket.timeout, urllib.error.URLError) as e:
            print(f"❌ Server health check failed: {str(e)}")
            modal.experimental.stop_fetching_inputs()
            raise Exception("ComfyUI server is not healthy, stopping container")

# Example usage:
# 1. Deploy: modal deploy modal_reactor_app.py
# 2. Interactive UI: modal serve modal_reactor_app.py
# 3. API call: POST to the endpoint with source_image and target_video files 
