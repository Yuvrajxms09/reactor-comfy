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
reactor_models_vol = modal.Volume.from_name("Reactor-comfy-files", create_if_missing=False)

# Build the Modal image with minimal dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git", 
        "ffmpeg",  # Required for video processing
        "libgl1-mesa-glx",  # OpenCV dependencies
        "libglib2.0-0",
        "curl"  # Required for downloading insightface wheel
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
        "pip install onnxruntime-gpu torch torchvision torchaudio"
    )
    .run_commands(
        # Install additional dependencies that are commonly missing
        "pip install opencv-python-headless numpy scipy scikit-image",
        # Ensure all dependencies are properly installed
        "pip install --upgrade pip setuptools wheel"
    )
    .run_commands(
        # Install the correct insightface wheel for Python 3.11 on Linux
        "curl -L -o insightface-0.7.3-cp311-cp311-linux_x86_64.whl https://github.com/deepinsight/insightface/releases/download/v0.7.3/insightface-0.7.3-cp311-cp311-linux_x86_64.whl || echo 'Wheel download failed, using PyPI version'",
        "pip install insightface-0.7.3-cp311-cp311-linux_x86_64.whl || pip install insightface==0.7.3",
        "rm -f insightface-0.7.3-cp311-cp311-linux_x86_64.whl"
    )
    .run_commands(
        # Initialize custom nodes properly
        "cd /root/comfy/ComfyUI/custom_nodes/ComfyUI-ReActor && python install.py"
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
    model_mappings = [
        ("/models/insightface/inswapper_128.onnx", f"{comfyui_base}/models/insightface/inswapper_128.onnx"),
        ("/models/insightface/inswapper_128_fp16.onnx", f"{comfyui_base}/models/insightface/inswapper_128_fp16.onnx"),
        ("/models/ultralytics/bbox/face_yolov8m.pt", f"{comfyui_base}/models/ultralytics/bbox/face_yolov8m.pt"),
        ("/models/sams/sam_vit_l_0b3195.pth", f"{comfyui_base}/models/sams/sam_vit_l_0b3195.pth"),
        ("/models/facerestore_models/GFPGANv1.4.pth", f"{comfyui_base}/models/facerestore_models/GFPGANv1.4.pth"),
        ("/models/facerestore_models/codeformer-v0.1.0.pth", f"{comfyui_base}/models/facerestore_models/codeformer-v0.1.0.pth"),
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
    
    # Test custom node imports
    print("🔍 Testing custom node imports...")
    try:
        import sys
        sys.path.insert(0, comfy_dir)
        sys.path.insert(0, reactor_dir)
        sys.path.insert(0, vhs_dir)
        
        # Test ReActor import
        try:
            from ComfyUI_ReActor import nodes
            print("✅ ReActor nodes imported successfully")
        except Exception as e:
            print(f"❌ Failed to import ReActor nodes: {str(e)}")
            # Try alternative import
            try:
                import nodes
                print("✅ ReActor nodes imported successfully (alternative)")
            except Exception as e2:
                print(f"❌ Alternative ReActor import failed: {str(e2)}")
        
        # Test VHS import
        try:
            from ComfyUI_VideoHelperSuite import nodes
            print("✅ VHS nodes imported successfully")
        except Exception as e:
            print(f"❌ Failed to import VHS nodes: {str(e)}")
            # Try alternative import
            try:
                import nodes
                print("✅ VHS nodes imported successfully (alternative)")
            except Exception as e2:
                print(f"❌ Alternative VHS import failed: {str(e2)}")
        
    except Exception as e:
        print(f"❌ Error testing imports: {str(e)}")
    
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
        # Set up custom nodes before launching ComfyUI
        self.setup_custom_nodes()
        
        cmd = f"comfy launch --background -- --port {self.port}"
        subprocess.run(cmd, shell=True, check=True)
        print("✅ ComfyUI server launched")

    def setup_custom_nodes(self):
        """Ensure custom nodes are properly set up for ComfyUI"""
        comfy_dir = "/root/comfy/ComfyUI"
        
        # Add custom nodes to Python path
        import sys
        sys.path.insert(0, comfy_dir)
        sys.path.insert(0, f"{comfy_dir}/custom_nodes/ComfyUI-ReActor")
        sys.path.insert(0, f"{comfy_dir}/custom_nodes/ComfyUI-VideoHelperSuite")
        
        # Create __init__.py files if they don't exist
        reactor_dir = f"{comfy_dir}/custom_nodes/ComfyUI-ReActor"
        vhs_dir = f"{comfy_dir}/custom_nodes/ComfyUI-VideoHelperSuite"
        
        for node_dir in [reactor_dir, vhs_dir]:
            if os.path.exists(node_dir):
                init_file = os.path.join(node_dir, "__init__.py")
                if not os.path.exists(init_file):
                    with open(init_file, 'w') as f:
                        f.write("# Custom node package\n")
                    print(f"✅ Created __init__.py for {os.path.basename(node_dir)}")
        
        print("✅ Custom nodes setup completed")

    @modal.method()
    def infer(self, workflow_path: str = "/root/face-video-swapv2.json"):
        """Run inference on a workflow"""
        print("🚀 Starting inference process...")
        print(f"🔍 DEBUG: Workflow path: {workflow_path}")
        
        # Check server health
        print("🔍 Checking ComfyUI server health...")
        self.poll_server_health()
        
        print(f"📁 Executing workflow: {workflow_path}")
        
        # Check if workflow file exists
        if not os.path.exists(workflow_path):
            print(f"❌ Workflow file not found: {workflow_path}")
            raise Exception(f"Workflow file not found: {workflow_path}")
        
        # Read and validate workflow
        try:
            with open(workflow_path, 'r') as f:
                workflow_data = json.load(f)
            print(f"✅ Workflow loaded successfully with {len(workflow_data)} nodes")
            
            # Validate workflow structure
            required_nodes = ["1", "4", "5", "6"]  # VHS_LoadVideoPath, ReActorFaceSwap, LoadImage, VHS_VideoCombine
            missing_nodes = [node_id for node_id in required_nodes if node_id not in workflow_data]
            if missing_nodes:
                raise Exception(f"Missing required nodes in workflow: {missing_nodes}")
            
            # Check node types
            node_types = {
                "1": "VHS_LoadVideoPath",
                "4": "ReActorFaceSwap", 
                "5": "LoadImage",
                "6": "VHS_VideoCombine"
            }
            
            for node_id, expected_type in node_types.items():
                actual_type = workflow_data[node_id].get("class_type", "unknown")
                if actual_type != expected_type:
                    print(f"⚠️ Node {node_id} has type '{actual_type}', expected '{expected_type}'")
                else:
                    print(f"✅ Node {node_id} ({expected_type}) validated")
            
        except Exception as e:
            print(f"❌ Failed to load workflow: {str(e)}")
            raise Exception(f"Failed to load workflow: {str(e)}")
        
        # Clear previous outputs before running
        output_dir = "/root/comfy/ComfyUI/output"
        temp_dir = "/root/comfy/ComfyUI/temp"
        
        print("🧹 Clearing previous outputs...")
        for directory in [output_dir, temp_dir]:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    file_path = os.path.join(directory, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"🗑️ Removed old file: {file}")
        
        # Set up environment for ComfyUI
        env = dict(os.environ)
        comfy_dir = "/root/comfy/ComfyUI"
        env['PYTHONPATH'] = f"{comfy_dir}:{comfy_dir}/custom_nodes/ComfyUI-ReActor:{comfy_dir}/custom_nodes/ComfyUI-VideoHelperSuite:{env.get('PYTHONPATH', '')}"
        
        # Run the workflow
        try:
            cmd = f"comfy run --workflow {workflow_path} --wait --timeout 1800 --verbose"
            print(f"🚀 Running command: {cmd}")
            
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
            
            # Print detailed output for debugging
            if result.stdout:
                print("📋 STDOUT:")
                print(result.stdout)
            
            if result.stderr:
                print("⚠️ STDERR:")
                print(result.stderr)
            
            if result.returncode != 0:
                print(f"❌ Command failed:")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                raise Exception(f"Command failed: {result.stderr}")
            
            print("✅ Workflow executed successfully")
            
        except subprocess.TimeoutExpired:
            print("⏰ Workflow execution timed out after 30 minutes")
            raise Exception("Workflow execution timed out after 30 minutes")
        except Exception as e:
            print(f"❌ Error running workflow: {str(e)}")
            raise Exception(f"Error running workflow: {str(e)}")
        
        # Collect outputs with better debugging
        print("📦 Collecting outputs...")
        outputs = []
        
        # Check output directory for images and videos
        if os.path.exists(output_dir):
            print(f"🔍 Checking output directory: {output_dir}")
            files = list(Path(output_dir).iterdir())
            for f in files:
                if f.is_file():
                    file_size = f.stat().st_size
                    print(f"📄 Found output file: {f.name} ({file_size} bytes)")
                    
                    # Check if it's a video file
                    if f.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                        print(f"🎥 Found video file in output: {f.name} ({file_size} bytes)")
                        outputs.append({
                            "type": "video",
                            "name": f.name,
                            "data": f.read_bytes()
                        })
                    else:
                        outputs.append({
                            "type": "output",
                            "name": f.name,
                            "data": f.read_bytes()
                        })
        
        # Check temp directory for additional files
        if os.path.exists(temp_dir):
            print(f"🔍 Checking temp directory: {temp_dir}")
            files = list(Path(temp_dir).iterdir())
            for f in files:
                if f.is_file():
                    file_size = f.stat().st_size
                    print(f"📄 Found temp file: {f.name} ({file_size} bytes)")
                    
                    # Only add non-video files from temp (videos should be in output)
                    if f.suffix.lower() not in ['.mp4', '.avi', '.mov', '.mkv']:
                        outputs.append({
                            "type": "output",
                            "name": f.name,
                            "data": f.read_bytes()
                        })
                    else:
                        print(f"🎥 Found video in temp (will use output version): {f.name}")
        
        if not outputs:
            print("⚠️ No outputs found")
            # List all directories to help debug
            print("🔍 Directory contents:")
            for directory in [output_dir, temp_dir]:
                if os.path.exists(directory):
                    print(f"  {directory}: {os.listdir(directory)}")
                else:
                    print(f"  {directory}: Directory does not exist")
        else:
            print(f"✅ Found {len(outputs)} outputs")
            for output in outputs:
                print(f"  - {output['type']}: {output['name']} ({len(output['data'])} bytes)")
        
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
        import filetype
        from PIL import Image
        import io
        import cv2

        try:
            # File validation
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
                    temp_path = 'temp_validate.mp4'
                    with open(temp_path, 'wb') as f:
                        f.write(video_bytes)
                    
                    cap = cv2.VideoCapture(temp_path)
                    if not cap.isOpened():
                        raise Exception("Video could not be opened with OpenCV")
                    
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        raise Exception("Could not read first frame")
                    
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
                
                # Save uploaded files
                source_image_path = temp_dir_path / "source_face.jpg"
                target_video_path = temp_dir_path / "target_video.mp4"
                
                with open(source_image_path, "wb") as f:
                    f.write(source_image)
                
                with open(target_video_path, "wb") as f:
                    f.write(target_video)
                
                # Check if workflow file exists
                workflow_file = Path(__file__).parent / "face-video-swapv2.json"
                if not workflow_file.exists():
                    raise HTTPException(
                        status_code=500, 
                        detail="Workflow file 'face-video-swapv2.json' not found"
                    )
                
                # Load the workflow
                workflow_data = json.loads(workflow_file.read_text())

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

                # Return results with better debugging
                print(f"🔍 Processing {len(outputs)} outputs from inference...")
                
                if outputs:
                    # Look for video output first
                    video_outputs = [o for o in outputs if o["type"] == "video"]
                    if video_outputs:
                        video_output = video_outputs[0]
                        print(f"🎥 Returning video: {video_output['name']} ({len(video_output['data'])} bytes)")
                        return Response(
                            content=video_output["data"], 
                            media_type="video/mp4",
                            headers={
                                "Content-Disposition": f"attachment; filename={video_output['name']}",
                                "X-Client-ID": client_id,
                                "X-Output-Type": "video",
                                "X-Output-Size": str(len(video_output["data"])),
                                "X-Total-Outputs": str(len(outputs))
                            }
                        )
                    
                    # If no video, return first output with debug info
                    first_output = outputs[0]
                    print(f"📄 No video found, returning first output: {first_output['name']} ({len(first_output['data'])} bytes)")
                    
                    # Determine content type based on file extension
                    content_type = "application/octet-stream"
                    if first_output['name'].lower().endswith('.png'):
                        content_type = "image/png"
                    elif first_output['name'].lower().endswith('.jpg') or first_output['name'].lower().endswith('.jpeg'):
                        content_type = "image/jpeg"
                    elif first_output['name'].lower().endswith('.mp4'):
                        content_type = "video/mp4"
                    
                    return Response(
                        content=first_output["data"],
                        media_type=content_type,
                        headers={
                            "Content-Disposition": f"attachment; filename={first_output['name']}",
                            "X-Client-ID": client_id,
                            "X-Output-Type": first_output["type"],
                            "X-Output-Size": str(len(first_output["data"])),
                            "X-Total-Outputs": str(len(outputs))
                        }
                    )
                else:
                    print("❌ No outputs generated from workflow")
                    raise HTTPException(
                        status_code=500, 
                        detail="No output generated from workflow. Check the workflow configuration and input files."
                    )

        except HTTPException:
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
