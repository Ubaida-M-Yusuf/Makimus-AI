import os
import sys
import pickle
import json
import shutil
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from threading import Thread
from pathlib import Path
import time
import queue
import ctypes
import subprocess
import re
import gc
import warnings
from PIL import Image, ImageTk
import numpy as np

# Prevent PIL from crashing on legitimately large images (scanned maps, panoramas, etc.)
# Files that truly cannot be decoded still get caught by the try/except in open_image()
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# --- Cross-Platform Configuration & Auto-Tuning ---
def get_system_vram():
    """
    Cross-platform VRAM detection.
    Returns VRAM in bytes, or None if detection fails.
    """
    # Method 1: PyTorch (Best for NVIDIA CUDA and macOS MPS)
    try:
        import torch
        
        # NVIDIA CUDA (Windows/Linux)
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory
            return vram
        
        # Apple Silicon MPS (macOS)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            import psutil
            return int(psutil.virtual_memory().total * 0.8)
    except Exception:
        pass

    # Method 2: Windows WMIC (AMD/NVIDIA on Windows)
    if os.name == 'nt':
        try:
            cmd = 'wmic path win32_VideoController get AdapterRAM'
            output = subprocess.check_output(cmd, shell=True).decode('utf-8', errors='ignore')
            values = [int(s) for s in re.findall(r'\d+', output)]
            if values:
                return max(values)
        except Exception:
            pass
    
    # Method 3: Linux sysfs (AMD GPUs)
    elif sys.platform.startswith('linux'):
        try:
            import glob
            vram_paths = glob.glob('/sys/class/drm/card*/device/mem_info_vram_total')
            if vram_paths:
                with open(vram_paths[0], 'r') as f:
                    return int(f.read().strip())
        except Exception:
            pass
    
    # Method 4: macOS system_profiler
    elif sys.platform == 'darwin':
        try:
            cmd = 'system_profiler SPDisplaysDataType'
            output = subprocess.check_output(cmd, shell=True).decode('utf-8')
            match = re.search(r'VRAM.*?(\d+)\s*(MB|GB)', output, re.IGNORECASE)
            if match:
                size = int(match.group(1))
                unit = match.group(2).upper()
                return size * (1024**3 if unit == 'GB' else 1024**2)
        except Exception:
            pass
    
    return None

def determine_batch_size(vram_bytes=None):
    if vram_bytes is None:
        vram_bytes = get_system_vram()
    if vram_bytes is None:
        print("[CONFIG] Could not detect VRAM. Defaulting to Batch Size: 16")
        return 16
    
    vram_gb = vram_bytes / (1024**3)
    print(f"[CONFIG] Detected VRAM: {vram_gb:.2f} GB")
    
    if vram_gb >= 31:
        return 384
    elif vram_gb >= 23:
        return 256
    elif vram_gb >= 19:
        return 192
    elif vram_gb >= 15:
        return 160
    elif vram_gb >= 11:
        return 128
    elif vram_gb >= 7:
        return 64
    elif vram_gb >= 4:
        return 32
    else:
        return 16

def determine_video_batch_size(vram_bytes=None):
    if vram_bytes is None:
        vram_bytes = get_system_vram()
    if vram_bytes is None:
        return 8
    vram_gb = vram_bytes / (1024**3)
    if vram_gb >= 31:
        return 64
    elif vram_gb >= 23:
        return 48
    elif vram_gb >= 19:
        return 32
    elif vram_gb >= 15:
        return 24
    elif vram_gb >= 11:
        return 16
    elif vram_gb >= 7:
        return 12
    elif vram_gb >= 4:
        return 8
    else:
        return 4

# Query VRAM once — both batch sizes share the same detection result
_VRAM_BYTES = get_system_vram()
BATCH_SIZE = determine_batch_size(_VRAM_BYTES)
print(f"[CONFIG] Selected Batch Size: {BATCH_SIZE}")
VIDEO_BATCH_SIZE = determine_video_batch_size(_VRAM_BYTES)
print(f"[CONFIG] Selected Video Batch Size: {VIDEO_BATCH_SIZE}")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif",
              ".cr2", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf", ".pef", ".sr2")
VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v",
              ".wmv", ".flv", ".ts", ".mpg", ".mpeg", ".3gp", ".vob")
VIDEO_FRAME_INTERVAL = 5  # minimum seconds between sampled frames
MAX_FRAMES_PER_VIDEO = 50  # cap frames per video — interval scales up for long videos
TOP_RESULTS = 30
THUMBNAIL_SIZE = (180, 180)
MAX_THUMBNAIL_CACHE = 2000  # Limit RAM usage - clear old thumbnails after this
CELL_WIDTH = 220
CELL_HEIGHT = 260
CACHE_PREFIX = ".clip_cache_"
CACHE_SUFFIX = ".pkl"

MODEL_NAME = "ViT-L-14"
MODEL_PRETRAINED = "laion2b_s32b_b82k"

# ─── ONNX Toggle ────────────────────────────────────────────────────────────
# Set USE_ONNX = True ONLY if PyTorch CUDA doesn't work on your GPU
# (e.g. early RTX 50-series Blackwell cards on PyTorch < 2.7)
# For most users PyTorch native CUDA is faster and uses less VRAM.
USE_ONNX = False
# ─────────────────────────────────────────────────────────────────────────────

BG = "#1e1e1e"
PANEL_BG = "#252526"
CARD_BG = "#2d2d30"
FG = "#e0e0e0"
ACCENT = "#4CAF50"
ACCENT_SECONDARY = "#3fa9f5"
DANGER = "#f44336"
ORANGE = "#ff9800"
BORDER = "#3c3c3c"

RAW_EXTS = (".cr2", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf", ".pef", ".sr2")

def get_safe_path(path):
    """Prepend Windows extended path prefix to handle paths longer than 260 chars."""
    if os.name == 'nt':
        path = os.path.normpath(path)
        if not path.startswith('\\\\?\\'):
            path = '\\\\?\\' + path
    return path

def open_image(path):
    """Open any image including RAW formats. Falls back gracefully if rawpy not installed."""
    safe_path = get_safe_path(path)
    # Check extension on original path (without prefix)
    if path.lower().endswith(RAW_EXTS):
        try:
            import rawpy
            with rawpy.imread(safe_path) as raw:
                rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)
            img = Image.fromarray(rgb)
        except ImportError:
            safe_print(f"[RAW] rawpy not installed, skipping {os.path.basename(path)}. Install with: pip install rawpy")
            return None
        except Exception as e:
            safe_print(f"[RAW] Failed to open {os.path.basename(path)}: {e}")
            return None
    else:
        try:
            # Open via a file handle so PIL never sees the \\?\ extended-path prefix.
            # PIL's format-sniffing uses Python's built-in open() internally, which on some
            # Pillow/Windows versions mishandles the \\?\ prefix and raises
            # "cannot identify image file" even for valid JPEGs/WEBPs.
            # Passing an already-open binary file object bypasses that entirely.
            with open(safe_path, 'rb') as fh:
                img = Image.open(fh)
                img.load()   # force full decode while file handle is still open
        except MemoryError:
            safe_print(f"[IMAGE] Skipping {os.path.basename(path)}: image too large for available RAM")
            return None
        except Exception as e:
            safe_print(f"[IMAGE] Failed to open {os.path.basename(path)}: {e}")
            return None

    if img.mode == 'P' and 'transparency' in img.info:
        img = img.convert("RGBA")

    if img.mode == 'RGBA':
        # Composite onto white background — better for CLIP than black (matches training data)
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])  # use alpha channel as mask
        img = background
    elif img.mode != 'RGB':
        img = img.convert("RGB")
    return img

def safe_print(text, end='\n'):
    try:
        print(text, end=end)
    except:
        pass

class HybridCLIPModel:
    """
    Cross-Platform Hybrid Model Wrapper
    """
    def __init__(self):
        import torch
        import open_clip
        if USE_ONNX:
            import onnxruntime as ort
        
        # 1. Determine Device
        self.device_name = "CPU"
        
        try:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.device_name = f"CUDA (GPU {torch.cuda.get_device_name(0)})"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                self.device_name = "Metal (Apple GPU)"
            elif os.name == 'nt':
                try:
                    import torch_directml
                    self.device = torch_directml.device()
                    self.device_name = "DirectML (Windows GPU)"
                except ImportError:
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device("cpu")
        except Exception:
            self.device = torch.device("cpu")
        
        safe_print(f"[MODEL] Using Device: {self.device_name}")

        # Enable TF32 on Ampere+ for free matmul speedup (ignored on non-CUDA)
        try:
            import torch
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

        # Detect best inference dtype for this GPU — no fallback needed, autocast handles it
        self.amp_dtype = None  # None = disabled (CPU, DirectML, old GPU)
        try:
            import torch
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability(0)
                if major >= 8:  # Ampere (30xx) and newer — full BF16/FP16 Tensor Cores
                    # BF16 preferred on Ada (40xx) and Blackwell (50xx), FP16 on Ampere (30xx)
                    if major >= 9 or (major == 8 and minor >= 9):
                        self.amp_dtype = torch.bfloat16  # Ada/Blackwell
                    else:
                        self.amp_dtype = torch.float16   # Ampere (RTX 30xx)
                    safe_print(f"[MODEL] Mixed precision enabled: {self.amp_dtype}")
                elif major >= 7:  # Turing/Volta (RTX 20xx, GTX 16xx) — FP16 works
                    self.amp_dtype = torch.float16
                    safe_print(f"[MODEL] Mixed precision enabled: {self.amp_dtype}")
                else:
                    safe_print(f"[MODEL] Mixed precision disabled (GPU too old)")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.amp_dtype = torch.float16  # Apple MPS supports FP16
                safe_print(f"[MODEL] Mixed precision enabled: {self.amp_dtype} (MPS)")
        except Exception:
            self.amp_dtype = None  # safe fallback

        # RTX 50-series (Blackwell) check — warn if PyTorch version is too old
        try:
            import torch
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability(0)
                if major >= 12:  # sm_120 = Blackwell (RTX 50-series)
                    import torch
                    pt_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
                    if pt_version < (2, 7):
                        safe_print(f"\n{'='*60}")
                        safe_print(f"[WARNING] RTX 50-series (Blackwell) detected!")
                        safe_print(f"[WARNING] Your PyTorch version ({torch.__version__}) may not fully support this GPU.")
                        safe_print(f"[WARNING] If you see CUDA errors, upgrade PyTorch:")
                        safe_print(f"[WARNING] pip install torch --index-url https://download.pytorch.org/whl/cu128")
                        safe_print(f"[WARNING] Or enable ONNX fallback: set USE_ONNX = True at top of script")
                        safe_print(f"{'='*60}\n")
                    else:
                        safe_print(f"[MODEL] RTX 50-series detected — PyTorch {torch.__version__} has native support ✓")
        except Exception:
            pass

        safe_print(f"[MODEL] Loading: {MODEL_NAME}")
        
        # 2. Load PyTorch Model
        model_loaded = False
        
        try:
            import huggingface_hub
            huggingface_hub.constants.HF_HUB_OFFLINE = True
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                MODEL_NAME, pretrained=MODEL_PRETRAINED
            )
            self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)
            safe_print(f"[MODEL] Loaded from local cache")
            model_loaded = True
        except Exception:
            safe_print(f"[MODEL] Cache not available, connecting to download...")
        
        if not model_loaded:
            try:
                import huggingface_hub
                huggingface_hub.constants.HF_HUB_OFFLINE = False
                os.environ["HF_HUB_OFFLINE"] = "0"
                os.environ["TRANSFORMERS_OFFLINE"] = "0"
                safe_print(f"[MODEL] Downloading {MODEL_NAME} (this may take a while)...")
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    MODEL_NAME, pretrained=MODEL_PRETRAINED
                )
                self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)
                safe_print(f"[MODEL] Download complete!")
            except Exception as e:
                safe_print(f"[MODEL] Download failed: {e}")
                raise
        
        # Move model to device — fall back to CPU if GPU transfer fails (e.g. TDR reset,
        # exclusive GPU lock, driver crash, or cudaErrorDevicesUnavailable)
        try:
            self.model = self.model.to(self.device).eval()
        except Exception as gpu_err:
            safe_print(f"[MODEL] ⚠ GPU transfer failed ({gpu_err}), falling back to CPU")
            safe_print(f"[MODEL] Tip: close other GPU-heavy apps (games, other ML tools) and restart")
            self.device = torch.device("cpu")
            self.device_name = "CPU (GPU fallback)"
            self.amp_dtype = None  # disable mixed precision on CPU
            self.model = self.model.to(self.device).eval()

        # 3. Setup ONNX Visual Encoder (only if USE_ONNX = True)
        if USE_ONNX:
            self.setup_onnx_encoder()
        else:
            # ONNX disabled — use pure PyTorch (faster, less VRAM)
            self.onnx_visual_path = None
            self.use_onnx_visual = False
            self.visual_session = None
            self.onnx_disabled = True
            safe_print(f"[MODEL] Using PyTorch native inference (ONNX disabled)")
            safe_print(f"[MODEL] Ready!\n")

    def setup_onnx_encoder(self):
        """Setup ONNX Visual Encoder with graceful fallback"""
        import torch
        import onnxruntime as ort
        
        # Initialize fallback state first
        self.onnx_visual_path = None
        self.use_onnx_visual = False
        self.visual_session = None
        self.onnx_disabled = False
        
        # Test if ONNX export is supported before attempting
        if not self._test_onnx_support():
            safe_print(f"[ONNX] Not supported on this system")
            safe_print(f"[ONNX] Using PyTorch (works perfectly)")
            self.onnx_disabled = True
            safe_print(f"[MODEL] Ready!\n")
            return
        
        # Setup ONNX Visual Encoder
        cache_dir = Path.home() / ".cache" / "onnx_clip"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.onnx_visual_path = cache_dir / f"{MODEL_NAME.replace('-', '_')}_visual.onnx"
        
        if not self.onnx_visual_path.exists():
            safe_print(f"[ONNX] Attempting visual encoder export...")
            
            try:
                dummy_image = torch.randn(1, 3, 224, 224).to(self.device)
                with torch.no_grad():
                    torch.onnx.export(
                        self.model.visual,
                        dummy_image,
                        self.onnx_visual_path,
                        input_names=['pixel_values'],
                        output_names=['image_embeds'],
                        dynamic_axes={'pixel_values': {0: 'batch'}},
                        opset_version=14,
                        do_constant_folding=True,
                        verbose=False
                    )
                safe_print(f"[ONNX] ✓ Export successful")
                
            except (Exception, SystemError, RuntimeError, KeyboardInterrupt) as e:
                # Catch all possible exceptions including C++ errors
                if self.onnx_visual_path and self.onnx_visual_path.exists():
                    try:
                        self.onnx_visual_path.unlink()
                    except:
                        pass
                
                safe_print(f"[ONNX] Export failed, using PyTorch")
                self.onnx_visual_path = None
                self.use_onnx_visual = False
                self.visual_session = None
                self.onnx_disabled = True
                safe_print(f"[MODEL] Ready!\n")
                return
        
        self._create_onnx_session()
        safe_print(f"[MODEL] Ready!\n")
    
    def _test_onnx_support(self):
        """Quick test if ONNX export is supported"""
        import torch
        
        # Disable ONNX on Linux by default to prevent segfaults
        # PyTorch still runs on GPU normally without ONNX
        if sys.platform.startswith('linux'):
            safe_print("[ONNX] Disabled on Linux to prevent segfaults (GPU still active via PyTorch)")
            return False
        
        return True
    
    def _create_onnx_session(self):
        """Create or recreate ONNX inference session"""
        import torch
        import onnxruntime as ort
        
        # Initialize to False first
        self.use_onnx_visual = False
        self.visual_session = None
        
        # If ONNX is disabled or path doesn't exist, skip
        if getattr(self, 'onnx_disabled', False) or not hasattr(self, 'onnx_visual_path') or self.onnx_visual_path is None:
            return
        
        if self.onnx_visual_path and self.onnx_visual_path.exists():
            # Check for corrupted/zero-byte ONNX file before attempting to load
            try:
                file_size = self.onnx_visual_path.stat().st_size
                if file_size < 1024:  # anything under 1KB is certainly corrupt
                    safe_print(f"[ONNX] Corrupted ONNX file detected ({file_size} bytes), deleting...")
                    self.onnx_visual_path.unlink()
                    self.use_onnx_visual = False
                    self.onnx_disabled = True
                    return
            except Exception:
                pass

            try:
                providers = []
                
                if torch.cuda.is_available():
                    providers.append('CUDAExecutionProvider')
                
                if sys.platform == 'darwin':
                    providers.append('CoreMLExecutionProvider')
                
                if os.name == 'nt':
                    providers.append('DmlExecutionProvider')
                
                providers.append('CPUExecutionProvider')
                
                self.visual_session = ort.InferenceSession(str(self.onnx_visual_path), providers=providers)
                self.use_onnx_visual = True
                active_provider = self.visual_session.get_providers()[0]
                safe_print(f"[ONNX] Visual encoder ready on {active_provider}")
            except Exception as e:
                safe_print(f"[ONNX] Failed to load, using PyTorch: {e}")
                self.use_onnx_visual = False
                self.visual_session = None
    
    def _destroy_onnx_session(self):
        """Destroy ONNX session to free VRAM (only if ONNX was used)"""
        if hasattr(self, 'visual_session') and self.visual_session is not None:
            try:
                # Delete the session object
                del self.visual_session
                self.visual_session = None
                self.use_onnx_visual = False
                
                # Force CUDA cleanup
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                safe_print("[ONNX] Session destroyed, VRAM freed")
            except Exception as e:
                safe_print(f"[ONNX] Cleanup warning: {e}")

    def preprocess_image_onnx(self, image):
        target_size = 224
        w, h = image.size
        scale = target_size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = image.resize((new_w, new_h), Image.BICUBIC)
        left = (new_w - target_size) // 2
        top = (new_h - target_size) // 2
        img = img.crop((left, top, left + target_size, top + target_size))
        img_np = np.array(img).astype(np.float32) / 255.0
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        img_np = (img_np - mean) / std
        img_np = img_np.transpose(2, 0, 1)
        return np.expand_dims(img_np, axis=0)
    
    def preprocess_image_pytorch(self, image):
        return self.preprocess(image).unsqueeze(0)
    
    def encode_image_batch(self, images):
        import torch
        
        # Only try ONNX if it's enabled and session exists
        if getattr(self, 'use_onnx_visual', False) and self.visual_session is not None:
            try:
                batch_inputs = [self.preprocess_image_onnx(img) for img in images]
                input_tensor = np.concatenate(batch_inputs, axis=0)
                outputs = self.visual_session.run(None, {"pixel_values": input_tensor})
                features = outputs[0]
                norms = np.linalg.norm(features, axis=1, keepdims=True)
                return features / norms
            except Exception as e:
                safe_print(f"[ONNX] Inference failed, falling back to PyTorch: {e}")
                # Disable ONNX for future calls
                self.use_onnx_visual = False
        
        # PyTorch path (always works, with optional mixed precision)
        try:
            batch_tensors = [self.preprocess_image_pytorch(img) for img in images]
            input_tensor = torch.cat(batch_tensors).to(self.device)
            amp_dtype = getattr(self, 'amp_dtype', None)
            with torch.no_grad():
                if amp_dtype is not None and torch.cuda.is_available():
                    with torch.autocast(device_type='cuda', dtype=amp_dtype):
                        features = self.model.encode_image(input_tensor)
                elif amp_dtype is not None and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    with torch.autocast(device_type='mps', dtype=amp_dtype):
                        features = self.model.encode_image(input_tensor)
                else:
                    features = self.model.encode_image(input_tensor)
                features = features.float()  # cast back to float32 for normalization
                features = features / features.norm(dim=-1, keepdim=True)
            
            result = features.cpu().numpy()
            del features, input_tensor, batch_tensors
            return result
        except Exception as e:
            safe_print(f"[ERROR] Image encoding failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def encode_tensor_batch(self, tensors):
        """Encode a batch of already-preprocessed tensors (torch.Tensor, shape [N,3,224,224]).

        Called by _process_batch when preprocessing was done in prefetch workers.
        Skips the serial preprocess_image_pytorch loop that blocked the GPU in encode_image_batch.
        encode_image_batch (PIL path) is unchanged — still used by video indexing and image search.
        """
        import torch
        try:
            stacked = torch.stack(tensors)
            # pin_memory allows async non-blocking CPU→GPU transfer on CUDA — GPU doesn't
            # stall waiting for the copy to finish, improving utilization significantly.
            if torch.cuda.is_available():
                stacked = stacked.pin_memory()
                input_tensor = stacked.to(self.device, non_blocking=True)
            else:
                input_tensor = stacked.to(self.device)
            amp_dtype = getattr(self, 'amp_dtype', None)
            with torch.no_grad():
                if amp_dtype is not None and torch.cuda.is_available():
                    with torch.autocast(device_type='cuda', dtype=amp_dtype):
                        features = self.model.encode_image(input_tensor)
                elif amp_dtype is not None and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    with torch.autocast(device_type='mps', dtype=amp_dtype):
                        features = self.model.encode_image(input_tensor)
                else:
                    features = self.model.encode_image(input_tensor)
                features = features.float()
                features = features / features.norm(dim=-1, keepdim=True)
            result = features.cpu().numpy()
            del features, input_tensor, tensors
            return result
        except Exception as e:
            safe_print(f"[ERROR] Tensor batch encoding failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def encode_text(self, texts):
        import torch
        
        try:
            tokens = self.tokenizer(texts)
            if isinstance(tokens, dict):
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
            else:
                tokens = tokens.to(self.device)
            amp_dtype = getattr(self, 'amp_dtype', None)
            with torch.no_grad():
                if amp_dtype is not None and torch.cuda.is_available():
                    with torch.autocast(device_type='cuda', dtype=amp_dtype):
                        if isinstance(tokens, dict):
                            features = self.model.encode_text(**tokens)
                        else:
                            features = self.model.encode_text(tokens)
                else:
                    if isinstance(tokens, dict):
                        features = self.model.encode_text(**tokens)
                    else:
                        features = self.model.encode_text(tokens)
                features = features.float()  # cast back to float32 for normalization
                features = features / features.norm(dim=-1, keepdim=True)
            
            result = features.cpu().numpy()
            del features, tokens
            safe_print(f"[ENCODE] Text encoded successfully, shape: {result.shape}")
            return result
        except Exception as e:
            safe_print(f"[ERROR] Text encoding failed: {e}")
            import traceback
            traceback.print_exc()
            raise

class ImageSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Makimus - AI Media Search")
        self.root.geometry("1400x900")
        self.root.configure(bg=BG)
        
        if os.name == 'nt':
            self.apply_dark_title_bar()
        
        self.folder = None
        self.cache_file = None
        self.image_paths = []  # NOW STORES RELATIVE PATHS
        self.image_embeddings = None
        self.thumbnail_images = {}
        self.selected_images = set()
        self.excluded_folders = set()

        # Video index — parallel to image index
        self.video_paths = []         # list of (rel_video_path_str, timestamp_float)
        self.video_embeddings = None  # numpy array (M, 512)
        self.video_cache_file = None
        self._pending_video_refresh = False

        # Pending batch accumulators — filled during indexing, flushed before save/search
        # Avoids O(N²) np.concatenate per batch for large collections
        self._pending_image_batches = []
        self._pending_video_batches = []

        # Result type filter (set in build_ui)
        self.show_images_var = None
        self.show_videos_var = None

        self.clip_model = None
        self.model_loading = False
        
        self.is_indexing = False
        self.stop_indexing = False
        self.is_stopping = False
        
        self.is_searching = False
        self.stop_search = False
        self.search_thread = None
        self.index_thread = None
        self.click_timer = None
        
        self.total_found = 0
        self.search_generation = 0
        self.render_cols = 1
        self.thumbnail_count = 0  # reliable counter for grid placement
        self.thumbnail_queue = queue.Queue()
        self.all_search_results = []   # stores ALL sorted results in memory
        self.show_more_offset = 0      # how many results currently displayed
        self._stored_all_results = []  # backup for pagination
        self._thumbnail_worker_thread = None  # tracks active thumbnail loader thread
        
        # Queue for pending actions after stop
        self.pending_action = None
        
        self.build_ui()
        Thread(target=self.load_model, daemon=True).start()

    def apply_dark_title_bar(self):
        try:
            self.root.update()
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            value = ctypes.c_int(1)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(value), ctypes.sizeof(value))
        except:
            pass

    def get_cache_filename(self):
        # Format: .clip_cache_ViT-L-14_LAION2B.pkl (preserves hyphens)
        pretrained_simple = "LAION2B" if "laion2b" in MODEL_PRETRAINED.lower() else MODEL_PRETRAINED.upper()
        cache_name = f".clip_cache_{MODEL_NAME}_{pretrained_simple}.pkl"
        return [cache_name]

    def get_video_cache_filename(self):
        pretrained_simple = "LAION2B" if "laion2b" in MODEL_PRETRAINED.lower() else MODEL_PRETRAINED.upper()
        return f".clip_cache_videos_{MODEL_NAME}_{pretrained_simple}.pkl"

    def get_exclusions_path(self):
        if not self.folder:
            return None
        return os.path.join(self.folder, ".clip_exclusions.json")

    def _is_excluded(self, rel_path):
        if not self.excluded_folders:
            return False
        normalized = rel_path.replace(os.sep, "/")
        return any(pattern in normalized for pattern in self.excluded_folders)

    def load_exclusions(self):
        path = self.get_exclusions_path()
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.excluded_folders = set(data.get("excluded_patterns", []))
                safe_print(f"[EXCLUSIONS] Loaded {len(self.excluded_folders)} pattern(s)")
            except Exception as e:
                safe_print(f"[EXCLUSIONS] Load error: {e}")
                self.excluded_folders = set()
        else:
            self.excluded_folders = set()

    def save_exclusions(self):
        path = self.get_exclusions_path()
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"excluded_patterns": sorted(self.excluded_folders)}, f, indent=2)
            safe_print(f"[EXCLUSIONS] Saved {len(self.excluded_folders)} pattern(s)")
        except Exception as e:
            safe_print(f"[EXCLUSIONS] Save error: {e}")

    def open_exclusions_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Folder Exclusions")
        dialog.geometry("480x380")
        dialog.configure(bg=BG)
        dialog.resizable(True, True)
        dialog.grab_set()

        tk.Label(
            dialog,
            text="Exclude images whose path contains any pattern below.\n"
                 "Case-sensitive substring match (e.g. 'nsfw', 'temp', 'backup').\n"
                 "Use forward slashes for folder separators (e.g. 'raw/originals').",
            bg=BG, fg=FG, font=("Segoe UI", 9), justify="left", wraplength=450
        ).pack(padx=12, pady=(10, 4), anchor="w")

        list_frame = tk.Frame(dialog, bg=BG)
        list_frame.pack(fill="both", expand=True, padx=12, pady=4)

        scrollbar = tk.Scrollbar(list_frame, bg=PANEL_BG)
        scrollbar.pack(side="right", fill="y")

        listbox = tk.Listbox(
            list_frame, bg=CARD_BG, fg=FG, selectbackground=ACCENT,
            font=("Segoe UI", 10), yscrollcommand=scrollbar.set,
            highlightthickness=0, relief="flat"
        )
        listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=listbox.yview)

        for pattern in sorted(self.excluded_folders):
            listbox.insert(tk.END, pattern)

        entry_frame = tk.Frame(dialog, bg=BG)
        entry_frame.pack(fill="x", padx=12, pady=4)

        entry = tk.Entry(
            entry_frame, bg=CARD_BG, fg=FG, insertbackground=FG,
            font=("Segoe UI", 10), relief="flat",
            highlightthickness=1, highlightbackground=BORDER, highlightcolor=ACCENT
        )
        entry.pack(side="left", fill="x", expand=True, padx=(0, 6))

        def add_pattern():
            pat = entry.get().strip()
            if not pat:
                return
            if pat not in self.excluded_folders:
                self.excluded_folders.add(pat)
                listbox.insert(tk.END, pat)
                self.save_exclusions()
            entry.delete(0, tk.END)

        def remove_pattern():
            sel = listbox.curselection()
            if not sel:
                return
            pat = listbox.get(sel[0])
            self.excluded_folders.discard(pat)
            listbox.delete(sel[0])
            self.save_exclusions()

        entry.bind("<Return>", lambda e: add_pattern())
        ttk.Button(entry_frame, text="Add", command=add_pattern, width=8).pack(side="left")

        btn_frame = tk.Frame(dialog, bg=BG)
        btn_frame.pack(fill="x", padx=12, pady=(0, 4))
        ttk.Button(btn_frame, text="Remove Selected", command=remove_pattern, width=16, style="Danger.TButton").pack(side="left", padx=(0, 8))
        ttk.Button(btn_frame, text="Close", command=dialog.destroy, width=10).pack(side="left")

        tk.Label(
            dialog,
            text="Note: Run Refresh after changing exclusions to apply them to the index.",
            bg=BG, fg=ORANGE, font=("Segoe UI", 8, "italic")
        ).pack(padx=12, pady=(0, 8), anchor="w")

    def load_model(self):
        self.model_loading = True
        self.root.after(0, lambda: self.update_status("Loading model...", "orange"))
        try:
            self.clip_model = HybridCLIPModel()
            self.root.after(0, lambda: self.update_status("Ready", "green"))
            safe_print(f"[LOAD] Success!\n")
        except Exception as e:
            safe_print(f"[ERROR] {e}")
            err_msg = str(e)
            self.root.after(0, lambda: self.update_status("Load Failed", "red"))
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load model\n{err_msg}"))
        self.model_loading = False

    def build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=PANEL_BG)
        style.configure("TLabel", background=PANEL_BG, foreground=FG)
        style.configure("TButton", background=ACCENT, foreground=FG, padding=6, borderwidth=0)
        style.map("TButton", background=[("active", "#5ecf60")])
        style.configure("Accent.TButton", background=ACCENT_SECONDARY, foreground=FG, padding=6, borderwidth=0)
        style.map("Accent.TButton", background=[("active", "#67c6ff")])
        style.configure("Danger.TButton", background=DANGER, foreground=FG, padding=6, borderwidth=0)
        style.map("Danger.TButton", background=[("active", "#ff6a6a")])
        style.configure("Horizontal.TProgressbar", troughcolor=PANEL_BG, background=ACCENT)
        style.configure("Vertical.TScrollbar", background=PANEL_BG, troughcolor=PANEL_BG, arrowcolor=FG)

        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x", padx=8, pady=6)
        
        self.btn_folder = ttk.Button(top, text="Folder", command=self.on_select_folder, width=10)
        self.btn_folder.pack(side="left", padx=4)
        
        self.btn_cache = ttk.Button(top, text="Load Cache", command=self.on_select_cache, width=12, style="Accent.TButton")
        self.btn_cache.pack(side="left", padx=4)
        
        self.btn_refresh = ttk.Button(top, text="Refresh", command=self.on_force_reindex, width=10)
        self.btn_refresh.pack(side="left", padx=4)

        self.btn_index_videos = ttk.Button(top, text="Index Videos", command=self.on_index_videos_click, width=13)
        self.btn_index_videos.pack(side="left", padx=4)

        self.btn_clear = ttk.Button(top, text="Clear Cache", command=self.on_delete_cache, width=12, style="Danger.TButton")
        self.btn_clear.pack(side="left", padx=4)
        
        ttk.Label(top, text=f"Using: {MODEL_NAME}", foreground=ACCENT_SECONDARY).pack(side="left", padx=(16, 4))
        
        self.btn_stop = ttk.Button(top, text="STOP INDEX", command=self.stop_indexing_process, width=12, style="Danger.TButton")
        self.btn_stop.pack(side="left", padx=(20, 4))
        
        ttk.Button(top, text="EXIT", command=self.force_quit, width=12, style="Danger.TButton").pack(side="left", padx=6)

        ttk.Button(top, text="Exclusions", command=self.open_exclusions_dialog, width=12).pack(side="left", padx=6)

        self.status_label = ttk.Label(top, text="Starting...", width=35, anchor="w")
        self.status_label.pack(side="left", padx=10)
        self.stats_label = ttk.Label(top, text="")
        self.stats_label.pack(side="left")

        ttk.Button(top, text="?", command=self.show_index_info, width=3).pack(side="right", padx=(4, 0))

        search_frame = ttk.Frame(self.root, padding=8)
        search_frame.pack(fill="x", padx=8, pady=4)
        ttk.Label(search_frame, text="Search:").pack(side="left", padx=(0, 6))
        
        self.query_entry = tk.Entry(search_frame, font=("Segoe UI", 12), bg=CARD_BG, fg=FG, 
                                    insertbackground=FG, relief="flat", highlightthickness=1, 
                                    highlightcolor=ACCENT, highlightbackground=BORDER)
        self.query_entry.pack(side="left", fill="x", expand=True, padx=6)
        self.query_entry.bind("<Return>", lambda e: self.on_search_click())
        
        ttk.Button(search_frame, text="Search", command=self.on_search_click, width=12, style="Accent.TButton").pack(side="left", padx=4)
        ttk.Button(search_frame, text="Image", command=self.on_image_click, width=10).pack(side="left", padx=4)

        ctrl_frame = ttk.Frame(self.root, padding=8)
        ctrl_frame.pack(fill="x", padx=8, pady=4)
        
        ttk.Label(ctrl_frame, text="Similarity Score:").pack(side="left", padx=(0, 4))
        self.score_var = tk.DoubleVar(value=0.15)
        tk.Scale(ctrl_frame, from_=0.0, to=1.0, resolution=0.01, orient="horizontal",
                 variable=self.score_var, length=140, bg=PANEL_BG, fg=FG, troughcolor=BORDER, highlightthickness=0).pack(side="left")

        ttk.Label(ctrl_frame, text="Results Per Page:").pack(side="left", padx=(16, 4))
        self.top_n_var = tk.IntVar(value=TOP_RESULTS)
        tk.Scale(ctrl_frame, from_=10, to=500, resolution=10, orient="horizontal", 
                 variable=self.top_n_var, length=170, bg=PANEL_BG, fg=FG, troughcolor=BORDER, highlightthickness=0).pack(side="left")
        
        ttk.Button(ctrl_frame, text="Clear Results", command=self.on_clear_click, width=12).pack(side="left", padx=14)
        ttk.Button(ctrl_frame, text="Copy", command=self.on_copy_click, width=8).pack(side="left", padx=4)
        ttk.Button(ctrl_frame, text="Move", command=self.on_move_click, width=8, style="Accent.TButton").pack(side="left", padx=4)
        ttk.Button(ctrl_frame, text="Delete", command=self.on_delete_click, width=8, style="Danger.TButton").pack(side="left", padx=4)
        ttk.Button(ctrl_frame, text="Select All", command=self._select_all_cards, width=10).pack(side="left", padx=4)
        ttk.Button(ctrl_frame, text="Deselect All", command=self._deselect_all_cards, width=10).pack(side="left", padx=4)

        ttk.Separator(ctrl_frame, orient="vertical").pack(side="left", fill="y", padx=(14, 8), pady=4)
        self.show_images_var = tk.BooleanVar(value=True)
        self.show_videos_var = tk.BooleanVar(value=True)
        tk.Checkbutton(ctrl_frame, text="Images", variable=self.show_images_var,
                       bg=PANEL_BG, fg=FG, selectcolor=BG, activebackground=PANEL_BG,
                       font=("Segoe UI", 9)).pack(side="left", padx=(0, 4))
        tk.Checkbutton(ctrl_frame, text="Videos", variable=self.show_videos_var,
                       bg=PANEL_BG, fg=FG, selectcolor=BG, activebackground=PANEL_BG,
                       font=("Segoe UI", 9)).pack(side="left", padx=(0, 4))

        ttk.Separator(ctrl_frame, orient="vertical").pack(side="left", fill="y", padx=(8, 8), pady=4)
        self.dedup_video_var = tk.BooleanVar(value=False)
        tk.Checkbutton(ctrl_frame, text="Best frame/video", variable=self.dedup_video_var,
                       bg=PANEL_BG, fg=FG, selectcolor=BG, activebackground=PANEL_BG,
                       font=("Segoe UI", 9)).pack(side="left", padx=(0, 4))

        self.progress = ttk.Progressbar(self.root, mode='determinate')
        self.progress.pack(fill="x", padx=10, pady=6)
        self.progress_label = ttk.Label(self.root, text="", anchor="center")
        self.progress_label.pack()

        # Page navigation frame — packed BEFORE results_container so it stays at bottom
        self.show_more_frame = tk.Frame(self.root, bg=BG)
        # Inner frame to center buttons
        nav_inner = tk.Frame(self.show_more_frame, bg=BG)
        nav_inner.pack(anchor="center")
        self.prev_page_btn = ttk.Button(nav_inner, text="← Prev Page", command=self.prev_page_results, width=14, style="Accent.TButton")
        self.prev_page_btn.pack(side="left", padx=8, pady=6)
        self.page_label = ttk.Label(nav_inner, text="", background=BG, foreground=FG, font=("Segoe UI", 10))
        self.page_label.pack(side="left", padx=16)
        self.show_more_btn = ttk.Button(nav_inner, text="Next Page →", command=self.show_more_results, width=14, style="Accent.TButton")
        self.show_more_btn.pack(side="left", padx=8, pady=6)
        # show_more_frame only packed when there are results

        results_container = ttk.Frame(self.root, padding=6)
        results_container.pack(fill="both", expand=True, padx=8, pady=6)
        
        self.canvas = tk.Canvas(results_container, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(results_container, orient="vertical", command=self.canvas.yview)
        
        self.results_frame = tk.Frame(self.canvas, bg=BG, highlightthickness=0)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.results_frame, anchor="nw")
        
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.bind('<Configure>', self.on_canvas_configure)
        self.results_frame.bind('<Configure>', self._on_results_frame_configure)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Setup rubber-band selection on canvas
        self._setup_rubber_band()

        if sys.platform == 'darwin':
            self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-1 * e.delta), "units"))
        elif sys.platform.startswith('linux'):
            self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))
            self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))
        else:
            self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        # Intercept X button — warn user if indexing is running
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def is_safe_to_act(self, action_callback=None, action_name="action"):
        """
        Returns True if no indexing is happening.
        Shows a wait popup if indexing is running — never auto-stops.
        Only the STOP INDEX button stops indexing.
        """
        if self.is_indexing or self.is_stopping:
            messagebox.showinfo(
                "Indexing in Progress",
                "Indexing is currently running.\n\n"
                "Please wait for it to finish, or press STOP INDEX to save your partial cache and stop.\n\n"
                "(Your partial index is safe - stopping saves everything indexed so far.)"
            )
            return False
        return True

    def stop_all_processes(self):
        self.cancel_search(clear_ui=True)
        if self.is_indexing:
            self.stop_indexing_process()

    def on_select_folder(self):
        if not self.is_safe_to_act(action_callback=self.select_folder, action_name="select folder"):
            return
        self.cancel_search(clear_ui=True)
        self.select_folder()

    def on_select_cache(self):
        if not self.is_safe_to_act(action_callback=self.select_cache, action_name="load cache"):
            return
        self.cancel_search(clear_ui=True)
        self.select_cache()

    def on_force_reindex(self):
        if not self.is_safe_to_act(action_callback=self.force_reindex, action_name="refresh index"):
            return
        self.cancel_search(clear_ui=True)
        self.force_reindex()

    def on_delete_cache(self):
        if not self.is_safe_to_act(action_callback=self.delete_cache, action_name="clear cache"):
            return
        self.cancel_search(clear_ui=True)
        self.delete_cache()

    def on_clear_click(self):
        self.cancel_search(clear_ui=True)
        self.clear_results()
        self.update_status("Results cleared", "green")

    def on_copy_click(self):
        self.export_selected()

    def on_move_click(self):
        self.move_selected()

    def on_delete_click(self):
        self.delete_selected()

    def on_search_click(self):
        self.cancel_search(clear_ui=True)
        self.do_search()

    def on_image_click(self):
        # Image search does matrix multiply against image_embeddings
        # which the indexing thread is actively growing — unsafe to run concurrently
        if not self.is_safe_to_act(action_name="image search"):
            return
        self.cancel_search(clear_ui=True)
        self.image_search()

    def _on_results_frame_configure(self, event):
        """Update scrollregion after forcing Tkinter to finish all pending geometry calculations"""
        self.results_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)
        # Recalculate columns and reflow grid if column count changed
        new_cols = max(1, event.width // CELL_WIDTH)
        if new_cols != self.render_cols:
            self.render_cols = new_cols
            # Re-grid all existing thumbnail cards to new column layout
            children = self.results_frame.winfo_children()
            for idx, widget in enumerate(children):
                r, c = divmod(idx, new_cols)
                widget.grid(row=r, column=c, padx=6, pady=6)
            # Update counter to match actual children
            self.thumbnail_count = len(children)
            # Force scrollregion update after reflow
            self.results_frame.update_idletasks()
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def force_quit(self):
        if messagebox.askyesno("Force Quit", "Force quit application?"):
            os._exit(0)

    def on_close(self):
        """Called when user clicks the X button (WM_DELETE_WINDOW).
        If indexing is running, offer three choices instead of silently killing the process."""
        if not self.is_indexing and not self.is_stopping:
            self.root.destroy()
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Indexing in Progress")
        dialog.resizable(False, False)
        dialog.grab_set()
        dialog.configure(bg=PANEL_BG)

        # Centre over main window
        self.root.update_idletasks()
        x = self.root.winfo_x() + self.root.winfo_width() // 2 - 220
        y = self.root.winfo_y() + self.root.winfo_height() // 2 - 90
        dialog.geometry(f"440x170+{x}+{y}")

        ttk.Label(
            dialog,
            text="⚠  Indexing is currently running.",
            font=("Segoe UI", 11, "bold"),
            background=PANEL_BG, foreground=ORANGE
        ).pack(pady=(18, 4))

        ttk.Label(
            dialog,
            text="What would you like to do?",
            background=PANEL_BG, foreground=FG
        ).pack(pady=(0, 14))

        btn_frame = tk.Frame(dialog, bg=PANEL_BG)
        btn_frame.pack()

        def stop_and_close():
            dialog.destroy()
            self.stop_indexing = True
            self.is_stopping = True
            self._safe_after(0, lambda: self.update_status("Stopping & saving before exit...", ORANGE))
            safe_print("\n[CLOSE] Stop & save requested.")
            def _wait():
                if self.is_indexing:
                    self._safe_after(200, _wait)
                else:
                    try:
                        self.root.destroy()
                    except Exception:
                        pass
            self._safe_after(200, _wait)

        def quit_anyway():
            dialog.destroy()
            os._exit(0)

        def cancel():
            dialog.destroy()

        ttk.Button(btn_frame, text="Stop & Save",  command=stop_and_close,
                   style="Accent.TButton",  width=14).pack(side="left", padx=6)
        ttk.Button(btn_frame, text="Quit Anyway",  command=quit_anyway,
                   style="Danger.TButton",  width=14).pack(side="left", padx=6)
        ttk.Button(btn_frame, text="Cancel",       command=cancel,
                   width=10).pack(side="left", padx=6)

    def stop_indexing_process(self):
        if self.is_indexing and not self.is_stopping:
            self.stop_indexing = True
            self.is_stopping = True
            self.update_status("Stopping... Please wait for file save.", DANGER)
            safe_print("\n[STOP] Stop signal sent. Waiting for batch to finish...")
        elif self.is_stopping:
            if self.pending_action:
                safe_print("[STOP] Clearing pending action...")
                self.pending_action = None
                self.update_status("Stopping... (Pending action cancelled)", DANGER)
                self.btn_stop.config(text="STOP INDEX")

    def cancel_search(self, clear_ui=False):
        """Cancel ongoing search and optionally clear UI"""
        self.search_generation += 1
        with self.thumbnail_queue.mutex:
            self.thumbnail_queue.queue.clear()
        
        self.stop_search = True
        self.total_found = 0
        
        if clear_ui:
            self.clear_results()  # This will free thumbnail RAM
        
        if not self.is_indexing:
            self.progress.stop()
            self.progress['value'] = 0
            self.progress_label.config(text="")
        
        if self.is_searching:
             self.update_status("Search Cancelled", "orange")
        self.is_searching = False

    def select_folder(self):
        if self.clip_model is None:
            messagebox.showwarning("Wait", "Model is still loading...")
            return
        
        # Show dialog FIRST — don't wipe anything until user confirms a new folder
        folder = filedialog.askdirectory()
        if not folder:
            self.update_status("No folder selected", "orange")
            return
        
        # User confirmed a new folder — now safe to wipe old data
        self.image_paths = []
        self.image_embeddings = None
        self.folder = None
        self.cache_file = None
        self.video_paths = []
        self.video_embeddings = None
        self.video_cache_file = None
        self.clear_results()
        self.update_stats()
        
        self.folder = folder
        self.excluded_folders = set()
        self.load_exclusions()
        safe_print(f"\n{'='*60}\n[FOLDER] {folder}")
        
        cache_files = self.get_cache_filename()
        found_cache = None
        for cache_name in cache_files:
            cache_path = os.path.join(folder, cache_name)
            if os.path.exists(cache_path):
                found_cache = cache_path
                safe_print(f"[CACHE] Found existing: {cache_name}")
                break
        
        # Check for video cache first so we know before showing any popup
        video_cache_name = self.get_video_cache_filename()
        video_cache_path = os.path.join(folder, video_cache_name)
        found_video_cache = os.path.exists(video_cache_path)

        if found_cache:
            self.cache_file = found_cache
            self.load_cache_data(found_cache)
            self.query_entry.delete(0, tk.END)
        else:
            safe_print("[CACHE] Image cache not found")
            if not found_video_cache:
                # No image cache AND no video cache — ask to index images
                if messagebox.askyesno("Index Folder?", f"No cache found for this folder.\n\nIndex images now?"):
                    self.cache_file = os.path.join(folder, cache_files[0])
                    self.start_indexing(mode="full")
                else:
                    self.update_status("Folder loaded (Not indexed)", "orange")
            # If video cache exists, skip the popup — user has something useful already

        if found_video_cache:
            self.load_video_cache_data(video_cache_path)
            safe_print(f"[VCACHE] Auto-loaded: {video_cache_name}")
            self.query_entry.delete(0, tk.END)
            # Update status to reflect what's actually loaded
            has_images = self.image_embeddings is not None and len(self.image_paths) > 0
            has_videos = self.video_embeddings is not None and len(self.video_paths) > 0
            if has_images and has_videos:
                n_imgs = len(self.image_paths)
                n_vids = len(set(vp for vp, _ in self.video_paths))
                self.update_status(f"Loaded {n_imgs:,} images + {n_vids:,} videos", "green")
            elif has_videos:
                n_vids = len(set(vp for vp, _ in self.video_paths))
                self.update_status(f"Loaded video cache: {n_vids:,} videos", "green")
            elif has_images:
                self.update_status(f"Loaded {len(self.image_paths):,} images", "green")

    def select_cache(self):
        cache = filedialog.askopenfilename(filetypes=[("Pickle", "*.pkl")])
        if not cache: return
        self.load_cache_data(cache)
        # Clear search bar — don't auto-search on cache load
        self.query_entry.delete(0, tk.END)

    def load_cache_data(self, cache_path):
        try:
            safe_print(f"[CACHE] Loading: {cache_path}")
            self.update_status("Loading cache from disk...", "orange")
            
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
                # New format: (relative_paths, embeddings)
                self.image_paths, self.image_embeddings = data
            
            if hasattr(self.image_embeddings, 'cpu'):
                self.image_embeddings = self.image_embeddings.cpu().numpy()
            
            self.cache_file = cache_path
            self.folder = os.path.dirname(cache_path)
            
            self.load_exclusions()
            self.update_stats()
            n_imgs = len(self.image_paths)
            n_vids = len(set(vp for vp, _ in self.video_paths)) if self.video_paths else 0
            if n_vids > 0:
                self.update_status(f"Loaded {n_imgs:,} images, {n_vids:,} videos", "green")
            else:
                self.update_status(f"Loaded {n_imgs:,} images", "green")
            safe_print(f"[CACHE] Success. {n_imgs:,} images (relative paths).")
        except Exception as e:
            messagebox.showerror("Error", f"Load failed: {e}")
            self.update_status("Cache load failed", "red")

    def load_video_cache_data(self, cache_path):
        try:
            safe_print(f"[VCACHE] Loading: {cache_path}")
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
                self.video_paths, self.video_embeddings = data

            if hasattr(self.video_embeddings, 'cpu'):
                self.video_embeddings = self.video_embeddings.cpu().numpy()

            self.video_cache_file = cache_path
            self.update_stats()
            n_videos = len(set(vp for vp, _ in self.video_paths))
            n_frames = len(self.video_paths)
            safe_print(f"[VCACHE] Loaded {n_frames:,} frames from {n_videos:,} videos.")
        except Exception as e:
            safe_print(f"[VCACHE] Load error: {e}")
            self.video_paths = []
            self.video_embeddings = None

    def start_indexing(self, mode="full"):
        # Guard against double-start — should never happen but belt-and-suspenders
        if self.is_indexing:
            safe_print(f"[INDEX] ⚠ start_indexing called while already indexing (mode={mode}), ignoring")
            return

        self.stop_indexing = False
        self.is_stopping = False
        self.pending_action = None
        self.update_status("Indexing...", "orange")
        self.btn_stop.config(text="STOP INDEX")
        
        if mode == "full":
            self.index_thread = Thread(target=self.index_all_images, daemon=True)
        elif mode == "refresh":
            self.index_thread = Thread(target=self.refresh_index, daemon=True)
        elif mode == "video_full":
            self.index_thread = Thread(target=self.index_all_videos, daemon=True)
        elif mode == "video_refresh":
            self.index_thread = Thread(target=self.refresh_video_index, daemon=True)
        else:
            safe_print(f"[INDEX] ⚠ Unknown mode: {mode}")
            return

        self.index_thread.start()

    def refresh_index(self):
        if not self.folder or self.clip_model is None: return
        self.is_indexing = True
        
        # Only recreate ONNX if it's not permanently disabled
        if self.clip_model and not getattr(self.clip_model, 'use_onnx_visual', False):
            if not getattr(self.clip_model, 'onnx_disabled', False):
                try:
                    self.clip_model._create_onnx_session()
                except:
                    pass  # Silently fall back to PyTorch
        
        safe_print("\n[SCAN] Scanning folder for changes...")
        self.root.after(0, lambda: self.update_status("Scanning folder...", "orange"))
        
        current_disk_files = set()
        new_files_to_add = []

        # O(1) membership lookups — image_paths is a list so 'in' is O(N) per check
        existing_paths_set = set(self.image_paths)
        
        # Build set of current disk files (relative paths)
        for root, _, files in os.walk(self.folder):
            if self.stop_indexing: break
            for f in files:
                if f.lower().endswith(IMAGE_EXTS):
                    abs_path = os.path.join(root, f)
                    rel_path = os.path.relpath(abs_path, self.folder)
                    if self._is_excluded(rel_path):
                        continue
                    current_disk_files.add(rel_path)
                    if rel_path not in existing_paths_set:
                        new_files_to_add.append(abs_path)  # Pass absolute for processing
        
        if self.stop_indexing:
            self._handle_stop()
            return

        safe_print("[SCAN] Pruning deleted/renamed files...")
        valid_indices = []
        pruned_paths = []
        
        for i, rel_path in enumerate(self.image_paths):
            if rel_path in current_disk_files:
                valid_indices.append(i)
                pruned_paths.append(rel_path)
        
        removed_count = len(self.image_paths) - len(valid_indices)
        
        if removed_count > 0:
            if self.image_embeddings is not None:
                self.image_embeddings = self.image_embeddings[valid_indices]
            self.image_paths = pruned_paths
            safe_print(f"[SCAN] Pruned {removed_count} stale entries.")
        
        if new_files_to_add:
            safe_print(f"[SCAN] Found {len(new_files_to_add)} new files.")
            self._process_batch(new_files_to_add, is_update=True)
        else:
            if removed_count > 0:
                self._save_cache(allow_shrink=True)
            
            self.is_indexing = False
            self.is_stopping = False
            safe_print("[SCAN] Index is up to date.")
            self._safe_after(0, lambda: self.update_status("Up to date", "green"))
            self._safe_after(0, self.update_stats)

    def index_all_images(self):
        if not self.folder or self.clip_model is None: return
        self.is_indexing = True
        
        # Only recreate ONNX if it's not permanently disabled
        if self.clip_model and not getattr(self.clip_model, 'use_onnx_visual', False):
            if not getattr(self.clip_model, 'onnx_disabled', False):
                try:
                    self.clip_model._create_onnx_session()
                except:
                    pass  # Silently fall back to PyTorch
        
        # Keep old data alive during scan — only wipe AFTER we have new data
        # This prevents losing existing index if stop is pressed early
        old_paths = self.image_paths[:]
        old_embeddings = self.image_embeddings.copy() if self.image_embeddings is not None else None
        self.image_paths = []
        self.image_embeddings = None
        
        all_images = []
        for root, _, files in os.walk(self.folder):
            if self.stop_indexing: break
            for f in files:
                if f.lower().endswith(IMAGE_EXTS):
                    abs_path = os.path.join(root, f)
                    rel_path = os.path.relpath(abs_path, self.folder)
                    if not self._is_excluded(rel_path):
                        all_images.append(abs_path)
        
        if self.stop_indexing:
            # Restore old data if stopped during scan
            self.image_paths = old_paths
            self.image_embeddings = old_embeddings
            self._handle_stop()
            return

        if not all_images:
            self.image_paths = old_paths
            self.image_embeddings = old_embeddings
            self.is_indexing = False
            self._safe_after(0, lambda: self.update_status("No images found", "orange"))
            return

        safe_print(f"[INDEX] Found {len(all_images)} images.")
        self._process_batch(all_images, is_update=False)

    def _process_batch(self, file_list, is_update=False):
        """Process images and store RELATIVE paths.

        Pipeline:
        - 8 worker threads each do: open_image() + CLIP preprocess() → tensor
          (preprocessing is stateless transforms — fully thread-safe)
        - 2 batches are always prefetched so the GPU never waits for CPU
        - GPU receives pre-built tensors via encode_tensor_batch() — no serial
          preprocess loop blocking before encode

        Timeline with 2-batch lookahead:
          Workers:  [load+preprocess N+1][load+preprocess N+2]
          GPU:      [.to(device) + encode N                  ]
        CPU and GPU work simultaneously instead of sequentially.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
        try:
            total = len(file_list)
            processed = 0
            existing_paths_set = set(self.image_paths)
            self._pending_image_batches = []
            import torch

            # Use ONNX-aware flag once to decide which worker path to use
            use_onnx = getattr(self.clip_model, 'use_onnx_visual', False) and \
                       getattr(self.clip_model, 'visual_session', None) is not None

            if use_onnx:
                # ONNX path: workers return (path, pil_image) — preprocessing stays in encode_image_batch
                def load_worker(abs_path):
                    try:
                        # Limit OpenMP threads per worker — without this each worker spawns
                        # a full PyTorch thread pool, causing 4 workers × 6 OMP threads = 24
                        # threads on a 12-thread CPU, saturating it completely.
                        torch.set_num_threads(1)
                        img = open_image(abs_path)
                        return (abs_path, img, None)   # None = no tensor
                    except Exception:
                        return (abs_path, None, None)
            else:
                # PyTorch path: workers do load + CLIP preprocess → tensor
                # preprocess() is a torchvision Compose of stateless transforms — thread-safe
                clip_preprocess = self.clip_model.preprocess
                def load_worker(abs_path):
                    try:
                        # Limit OpenMP threads per worker — without this each worker spawns
                        # a full PyTorch thread pool, causing 4 workers × 6 OMP threads = 24
                        # threads on a 12-thread CPU, saturating it completely.
                        torch.set_num_threads(1)
                        img = open_image(abs_path)
                        if img is None:
                            return (abs_path, None, None)
                        tensor = clip_preprocess(img)   # shape [3, 224, 224]
                        return (abs_path, img, tensor)
                    except Exception:
                        return (abs_path, None, None)

            batches = [file_list[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
            # 12 workers = one per CPU thread on Ryzen 5600. Each worker is capped at
            # 1 OMP thread via torch.set_num_threads(1) so no explosion — 12 clean threads total.
            executor = ThreadPoolExecutor(max_workers=12)

            def submit_batch(idx):
                """Submit all images in batch idx as individual futures."""
                if idx < len(batches):
                    return [executor.submit(load_worker, p) for p in batches[idx]]
                return []

            # Prefetch 2 batches ahead — workers run during GPU encode of current batch
            prefetch_queue = [submit_batch(0), submit_batch(1)]

            # 64 = sweet spot: GPU gets fed twice per batch (128/64=2 flushes).
            # Workers decode the second half while GPU runs the first — no zero gaps.
            # 128 was too large (one slow image stalls GPU), 32 was too small (too many round-trips).
            STREAM_CHUNK = 64

            for batch_idx, batch_paths in enumerate(batches):
                if self.stop_indexing:
                    safe_print("\n[INDEX] Stopping batch loop.")
                    break

                # Pop the front (batch_idx, already being loaded)
                current_futures = prefetch_queue.pop(0)

                # Push the next+2 batch into the back of the queue
                prefetch_queue.append(submit_batch(batch_idx + 2))

                # ── Streaming encode ─────────────────────────────────────────────────
                # Instead of waiting for ALL futures then encoding in one shot
                # (which leaves GPU idle while slowest image loads), we use as_completed
                # to encode in chunks of STREAM_CHUNK as workers finish.
                # GPU gets fed continuously; one slow image no longer stalls everything.
                buf_tensors   = []   # preprocessed tensors  (PyTorch path)
                buf_pils      = []   # PIL images            (ONNX path)
                buf_paths     = []   # corresponding abs paths

                def _flush_buf():
                    if not buf_paths or self.stop_indexing:
                        buf_tensors.clear(); buf_pils.clear(); buf_paths.clear()
                        return
                    try:
                        if buf_tensors and not use_onnx:
                            feats = self.clip_model.encode_tensor_batch(buf_tensors)
                        else:
                            feats = self.clip_model.encode_image_batch(buf_pils)
                        if feats is not None and feats.size > 0:
                            nf, np_ = [], []
                            for i2, ap in enumerate(buf_paths):
                                rp = os.path.relpath(ap, self.folder)
                                if rp not in existing_paths_set:
                                    np_.append(rp); nf.append(feats[i2])
                                    existing_paths_set.add(rp)
                            if np_:
                                self.image_paths.extend(np_)
                                self._pending_image_batches.append(np.array(nf))
                                # `processed` lives in the enclosing scope; update via list cell
                                _proc[0] += len(np_)
                    except Exception as enc_e:
                        safe_print(f"[ERROR] Stream encode chunk: {enc_e}")
                    buf_tensors.clear(); buf_pils.clear(); buf_paths.clear()

                _proc = [0]  # mutable cell so _flush_buf can update processed count

                for fut in (_as_completed(current_futures) if current_futures else []):
                    if self.stop_indexing:
                        break
                    try:
                        abs_path, img, tensor = fut.result()
                    except Exception:
                        continue
                    if img is None:
                        continue
                    buf_paths.append(abs_path)
                    if tensor is not None:
                        buf_tensors.append(tensor)
                    else:
                        buf_pils.append(img)
                    if len(buf_paths) >= STREAM_CHUNK:
                        _flush_buf()

                if buf_paths and not self.stop_indexing:
                    _flush_buf()

                processed += _proc[0]

                if self.stop_indexing:
                    break

                # Flush CUDA cache every 30 batches — empty_cache() is a GPU sync barrier
                # that can take 100-500 ms on Windows; calling it too often stalls the pipeline
                if batch_idx % 30 == 0 and batch_idx > 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        torch.mps.empty_cache()

                pct = (processed / total) * 100 if total > 0 else 0
                msg = f"{'Updating' if is_update else 'Indexing'}: {processed:,}/{total:,}"
                self._safe_after(0, lambda v=pct, m=msg: self.update_progress(v, m))
                safe_print(f"\r[INDEX] {msg}", end='')

            executor.shutdown(wait=False)
            safe_print("")

        finally:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                gc.collect()
            except Exception as cleanup_err:
                safe_print(f"[INDEX] VRAM cleanup warning (non-fatal): {cleanup_err}")

        self._flush_pending_batches()
        self._save_cache()
        self._handle_stop()

    def _process_video_batch(self, file_list, is_update=False):
        """Extract frames from videos in parallel, encode with CLIP, store (rel_path, timestamp) tuples.

        Pipeline: up to 3 worker threads each extract frames from different videos simultaneously
        (CPU-bound OpenCV I/O) while the main thread encodes available results on GPU.
        Workers do NO GPU calls — they only decode frames and return PIL images.
        This mirrors the image indexing prefetch pattern: CPU and GPU work in parallel.

        Timeout per cap.read() is 3 s (was 8 s). After 3 consecutive timeouts in one video,
        the worker abandons remaining frames for that video (broken file guard).
        """
        try:
            import cv2
        except ImportError:
            safe_print("[VINDEX ERROR] OpenCV not installed. Run: pip install opencv-python")
            self._safe_after(0, lambda: messagebox.showerror(
                "Missing Dependency",
                "OpenCV is required for video indexing.\n\nInstall it with:\n  pip install opencv-python"
            ))
            self._handle_video_stop()
            return

        try:
            import torch
        except ImportError:
            torch = None

        try:
            from concurrent.futures import ThreadPoolExecutor
            import threading as _threading

            # Suppress FFmpeg/h264 codec warnings globally before any VideoCapture opens.
            # "Missing reference picture", "co located POCs unavailable" etc. are harmless
            # artifacts of seeking to non-keyframes — they flood the console but don't
            # affect output. Must be set before VideoCapture() is constructed.
            import os as _os
            _os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
            try:
                import cv2 as _cv2pre
                _cv2pre.setLogLevel(0)   # 0 = LOG_LEVEL_SILENT
            except Exception:
                pass

            total = len(file_list)
            existing_set = set(self.video_paths) if is_update else set()
            self._pending_video_batches = []
            CHUNK_SIZE = max(VIDEO_BATCH_SIZE * 2, 32)
            VIDEO_PARALLEL = 1  # one decode worker — keeps CPU low while still pipelining

            def encode_chunk(frames, timestamps, _rel_path, _existing):
                """Encode one chunk of frames and append to index. GPU — called from main thread only."""
                if not frames or self.stop_indexing:
                    return
                try:
                    features = self.clip_model.encode_image_batch(frames)
                    if features is None or features.size == 0:
                        return
                    new_tuples = []
                    new_features = []
                    for j, ts in enumerate(timestamps):
                        tup = (_rel_path, ts)
                        if tup not in _existing:
                            new_tuples.append(tup)
                            new_features.append(features[j])
                            _existing.add(tup)
                    if new_tuples:
                        self.video_paths.extend(new_tuples)
                        self._pending_video_batches.append(np.array(new_features))
                except Exception as enc_err:
                    safe_print(f"[VINDEX ERROR] Encoding failed: {enc_err}")
                    if torch and torch.cuda.is_available():
                        torch.cuda.empty_cache()

            def extract_frames(abs_video_path):
                """Worker: read all sample frames from one video. CPU only — no GPU.
                Returns (rel_path, [(pil_img, ts), ...])."""
                rel_path = os.path.relpath(abs_video_path, self.folder)
                safe_print(f"[VINDEX] Analyzing: {os.path.basename(abs_video_path)}")
                frames = []
                cap = None
                # Redirect stderr once for the entire video — ffmpeg's C lib prints NAL
                # unit errors directly to fd 2, bypassing Python/OpenCV log levels entirely.
                # safe_print uses stdout (fd 1) so it is unaffected.
                _devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
                _old_stderr_fd = _os.dup(2)
                _os.dup2(_devnull_fd, 2)
                try:
                    cap = cv2.VideoCapture(abs_video_path)
                    if not cap.isOpened():
                        safe_print(f"[VINDEX] Cannot open: {abs_video_path}")
                        return (rel_path, frames)

                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

                    # CAP_PROP_FPS and CAP_PROP_FRAME_COUNT unreliable for VBR/MKV/WebM
                    if fps <= 0:
                        fps = 25.0
                        safe_print(f"[VINDEX] fps=0 for {os.path.basename(abs_video_path)}, assuming 25fps")
                    if total_frames_count <= 0:
                        duration_seconds = 60.0
                        safe_print(f"[VINDEX] frame_count=0 for {os.path.basename(abs_video_path)}, assuming 60s duration")
                    else:
                        duration_seconds = total_frames_count / fps

                    if duration_seconds <= 0:
                        cap.release()
                        return (rel_path, frames)

                    if duration_seconds < VIDEO_FRAME_INTERVAL:
                        frames_to_sample = {duration_seconds / 2.0}
                    else:
                        interval = max(VIDEO_FRAME_INTERVAL, duration_seconds / MAX_FRAMES_PER_VIDEO)
                        t = interval  # skip t=0 — almost always black intro/title card
                        frames_to_sample = set()
                        while t < duration_seconds:
                            frames_to_sample.add(round(t, 3))
                            t += interval
                        if not frames_to_sample:
                            frames_to_sample = {duration_seconds / 2.0}

                    best_frame = None
                    best_brightness = -1.0
                    all_skipped = True

                    # Keyframe-aware seeking — seek to nearest keyframe (cheap, no head thrash),
                    # then grab forward to exact target. Much faster than random seeking on HDD
                    # and avoids reading entire video like naive sequential approach.
                    for target_t in sorted(frames_to_sample):
                        if self.stop_indexing:
                            break
                        if is_update and (rel_path, target_t) in existing_set:
                            continue
                        # Seek to keyframe just before target — AVSEEK_FLAG_BACKWARD finds
                        # nearest keyframe without full decode, then read() decodes just that frame.
                        cap.set(cv2.CAP_PROP_POS_MSEC, max(0, (target_t - 2.0)) * 1000.0)
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            continue
                        try:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        except Exception:
                            del frame
                            continue
                        mean_brightness = float(frame_rgb[::8, ::8].mean())
                        if mean_brightness > best_brightness:
                            best_brightness = mean_brightness
                            best_frame = (Image.fromarray(frame_rgb.copy()), target_t)
                        if mean_brightness >= 10:
                            all_skipped = False
                            frames.append((Image.fromarray(frame_rgb), target_t))
                        del frame, frame_rgb

                    cap.release()

                    if all_skipped and best_frame is not None:
                        safe_print(f"[VINDEX] All frames black for {os.path.basename(abs_video_path)}, using least-dark frame at t={best_frame[1]:.1f}s")
                        return (rel_path, [best_frame])

                    return (rel_path, frames)

                except (MemoryError, Exception) as video_err:
                    safe_print(f"[VINDEX] Error extracting frames from {os.path.basename(abs_video_path)}: {video_err}")
                    try:
                        if cap:
                            cap.release()
                    except Exception:
                        pass
                    return (rel_path, [])
                finally:
                    # Always restore stderr — even if an exception escaped
                    _os.dup2(_old_stderr_fd, 2)
                    _os.close(_old_stderr_fd)
                    _os.close(_devnull_fd)
            # --- Submit videos in bounded batches to cap RAM usage ---
            # Submitting ALL futures at once causes completed futures to pile up in RAM
            # (each holding up to MAX_FRAMES_PER_VIDEO decoded PIL images).
            # VIDEO_PARALLEL*2 in-flight at a time keeps all workers busy while preventing
            # unbounded memory growth on large video collections.
            SUBMIT_BATCH = VIDEO_PARALLEL * 2  # 6 futures in-flight max
            file_idx = 0

            with ThreadPoolExecutor(max_workers=VIDEO_PARALLEL) as executor:
                for batch_start in range(0, total, SUBMIT_BATCH):
                    if self.stop_indexing:
                        safe_print("\n[VINDEX] Stopping video batch loop.")
                        break
                    batch_paths = file_list[batch_start:batch_start + SUBMIT_BATCH]
                    batch_futures = [executor.submit(extract_frames, p) for p in batch_paths]

                    for future in batch_futures:
                        if self.stop_indexing:
                            safe_print("\n[VINDEX] Stopping video batch loop.")
                            break

                        try:
                            rel_video_path, frame_list = future.result()
                        except Exception as fe:
                            safe_print(f"[VINDEX] Future error for file {file_idx}: {fe}")
                            file_idx += 1
                            continue

                        # Encode frame_list in chunks on GPU (main thread only)
                        chunk_frames = []
                        chunk_timestamps = []
                        for pil_img, ts in frame_list:
                            if self.stop_indexing:
                                break
                            chunk_frames.append(pil_img)
                            chunk_timestamps.append(ts)
                            if len(chunk_frames) >= CHUNK_SIZE:
                                encode_chunk(chunk_frames, chunk_timestamps, rel_video_path, existing_set)
                                chunk_frames = []
                                chunk_timestamps = []
                        if chunk_frames and not self.stop_indexing:
                            encode_chunk(chunk_frames, chunk_timestamps, rel_video_path, existing_set)

                        del frame_list

                        # Free GPU memory every 5 videos
                        if file_idx % 5 == 0:
                            if torch and torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            elif torch and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                                torch.mps.empty_cache()

                        # Periodic RAM cleanup every 10 videos
                        if file_idx % 10 == 0:
                            gc.collect()

                        file_idx += 1
                        pct = (file_idx / total) * 100
                        n_frames = len(self.video_paths)
                        msg = f"{'Updating' if is_update else 'Indexing'} Videos: {file_idx:,}/{total:,} files ({n_frames:,} frames)"
                        self._safe_after(0, lambda v=pct, m=msg: self.update_progress(v, m))
                        safe_print(f"\r[VINDEX] {msg}", end='')

            safe_print("")

        finally:
            try:
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                elif torch and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                gc.collect()
            except Exception as cleanup_err:
                safe_print(f"[VINDEX] VRAM cleanup warning (non-fatal): {cleanup_err}")

        self._flush_pending_batches()  # consolidate accumulated video batches before saving
        self._save_video_cache()
        self._handle_video_stop()

    def index_all_videos(self):
        if not self.folder or self.clip_model is None:
            return
        self.is_indexing = True

        if self.clip_model and not getattr(self.clip_model, 'use_onnx_visual', False):
            if not getattr(self.clip_model, 'onnx_disabled', False):
                try:
                    self.clip_model._create_onnx_session()
                except:
                    pass

        # Keep old data alive during scan — only wipe AFTER we have new data
        old_video_paths = self.video_paths[:]
        old_video_embeddings = self.video_embeddings.copy() if self.video_embeddings is not None else None
        self.video_paths = []
        self.video_embeddings = None

        all_videos = []
        for root_dir, _, files in os.walk(self.folder):
            if self.stop_indexing:
                break
            for f in files:
                if f.lower().endswith(VIDEO_EXTS):
                    abs_path = os.path.join(root_dir, f)
                    rel_path = os.path.relpath(abs_path, self.folder)
                    if not self._is_excluded(rel_path):
                        all_videos.append(abs_path)

        if self.stop_indexing:
            # Restore old data if stopped during scan
            self.video_paths = old_video_paths
            self.video_embeddings = old_video_embeddings
            self._handle_video_stop()
            return

        if not all_videos:
            self.video_paths = old_video_paths
            self.video_embeddings = old_video_embeddings
            self.is_indexing = False
            self._safe_after(0, lambda: self.update_status("No videos found", "orange"))
            return

        safe_print(f"[VINDEX] Found {len(all_videos)} video files.")
        self._process_video_batch(all_videos, is_update=False)

    def refresh_video_index(self):
        if not self.folder or self.clip_model is None:
            return
        self.is_indexing = True

        if self.clip_model and not getattr(self.clip_model, 'use_onnx_visual', False):
            if not getattr(self.clip_model, 'onnx_disabled', False):
                try:
                    self.clip_model._create_onnx_session()
                except:
                    pass

        safe_print("\n[VSCAN] Scanning folder for video changes...")
        self._safe_after(0, lambda: self.update_status("Scanning for video changes...", "orange"))

        current_disk_videos = set()
        new_videos_to_add = []

        for root_dir, _, files in os.walk(self.folder):
            if self.stop_indexing:
                break
            for f in files:
                if f.lower().endswith(VIDEO_EXTS):
                    abs_path = os.path.join(root_dir, f)
                    rel_path = os.path.relpath(abs_path, self.folder)
                    if self._is_excluded(rel_path):
                        continue
                    current_disk_videos.add(rel_path)
                    already_indexed = any(vp == rel_path for vp, _ in self.video_paths)
                    if not already_indexed:
                        new_videos_to_add.append(abs_path)

        if self.stop_indexing:
            self._handle_video_stop()
            return

        keep_indices = [i for i, (vp, _) in enumerate(self.video_paths) if vp in current_disk_videos]
        removed_count = len(self.video_paths) - len(keep_indices)

        if removed_count > 0:
            if self.video_embeddings is not None:
                self.video_embeddings = self.video_embeddings[keep_indices]
            self.video_paths = [self.video_paths[i] for i in keep_indices]
            safe_print(f"[VSCAN] Pruned {removed_count} stale frame entries.")

        if new_videos_to_add:
            safe_print(f"[VSCAN] Found {len(new_videos_to_add)} new video files.")
            self._process_video_batch(new_videos_to_add, is_update=True)
        else:
            if removed_count > 0:
                self._save_video_cache(allow_shrink=True)
            self.is_indexing = False
            self.is_stopping = False
            safe_print("[VSCAN] Video index is up to date.")
            self._safe_after(0, lambda: self.update_status("Video index up to date", "green"))
            self._safe_after(0, self.update_stats)

    def _flush_pending_batches(self):
        """Consolidate accumulated batch lists into single numpy arrays.
        Called before save and before search to avoid O(N²) concatenation during indexing."""
        if getattr(self, '_pending_image_batches', None):
            batches = self._pending_image_batches
            self._pending_image_batches = []
            stacked = np.vstack(batches)
            if self.image_embeddings is None:
                self.image_embeddings = stacked
            else:
                self.image_embeddings = np.vstack([self.image_embeddings, stacked])

        if getattr(self, '_pending_video_batches', None):
            batches = self._pending_video_batches
            self._pending_video_batches = []
            stacked = np.vstack(batches)
            if self.video_embeddings is None:
                self.video_embeddings = stacked
            else:
                self.video_embeddings = np.vstack([self.video_embeddings, stacked])

    def _save_cache(self, allow_shrink=False):
        """Save cache with RELATIVE paths — never overwrites a larger existing cache.
        Pass allow_shrink=True when a legitimate prune has reduced the count (e.g. refresh after delete)."""
        if self.image_embeddings is not None and len(self.image_paths) > 0:
            try:
                # Never overwrite a larger cache with a smaller one
                # This protects against partial index runs destroying good data.
                # Exception: allow_shrink=True when caller has already pruned stale entries.
                if not allow_shrink and os.path.exists(self.cache_file):
                    try:
                        with open(self.cache_file, "rb") as f:
                            existing_paths, _ = pickle.load(f)
                        if len(existing_paths) > len(self.image_paths):
                            safe_print(f"[CACHE] ⚠ Skipping save — existing cache has {len(existing_paths):,} images, current has only {len(self.image_paths):,}")
                            return
                    except Exception:
                        pass  # Can't read existing — proceed with save

                temp_file = self.cache_file + ".tmp"
                with open(temp_file, "wb") as f:
                    pickle.dump((self.image_paths, self.image_embeddings), f)
                if os.path.exists(self.cache_file):
                    os.remove(self.cache_file)
                os.rename(temp_file, self.cache_file)
                safe_print(f"[CACHE] Saved {len(self.image_paths)} relative paths to {self.cache_file}")
            except Exception as e:
                safe_print(f"[CACHE] Save Error: {e}")

    def _save_video_cache(self, allow_shrink=False):
        """Save video cache — never overwrites a larger existing cache.
        Pass allow_shrink=True when a legitimate prune has reduced the count."""
        if self.video_embeddings is not None and len(self.video_paths) > 0:
            try:
                # Never overwrite a larger cache with a smaller one
                if not allow_shrink and os.path.exists(self.video_cache_file):
                    try:
                        with open(self.video_cache_file, "rb") as f:
                            existing_paths, _ = pickle.load(f)
                        if len(existing_paths) > len(self.video_paths):
                            safe_print(f"[VCACHE] ⚠ Skipping save — existing cache has {len(existing_paths):,} frames, current has only {len(self.video_paths):,}")
                            return
                    except Exception:
                        pass  # Can't read existing — proceed with save

                temp_file = self.video_cache_file + ".tmp"
                with open(temp_file, "wb") as f:
                    pickle.dump((self.video_paths, self.video_embeddings), f)
                if os.path.exists(self.video_cache_file):
                    os.remove(self.video_cache_file)
                os.rename(temp_file, self.video_cache_file)
                safe_print(f"[VCACHE] Saved {len(self.video_paths):,} frame entries to {self.video_cache_file}")
            except Exception as e:
                safe_print(f"[VCACHE] Save Error: {e}")

    def _safe_after(self, ms, func):
        """root.after() wrapper safe to call from any thread even after window is destroyed.
        Silently ignored if the Tk interpreter is gone (e.g. during stop-and-close)."""
        try:
            self.root.after(ms, func)
        except Exception:
            pass

    def _handle_stop(self):
        was_stopped = self.stop_indexing
        count = len(self.image_paths)
        
        self.is_indexing = False
        self.stop_indexing = False
        self.is_stopping = False
        
        # Force VRAM release (only if ONNX was used)
        if self.clip_model and hasattr(self.clip_model, 'model'):
            import torch
            try:
                safe_print("[VRAM] Forcing memory release...")
                
                # Only destroy ONNX if it was actually being used
                if not getattr(self.clip_model, 'onnx_disabled', False):
                    self.clip_model._destroy_onnx_session()
                
                # PyTorch cleanup
                original_device = self.clip_model.device
                self.clip_model.model.cpu()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                
                gc.collect()
                
                # Move model back to GPU
                self.clip_model.model.to(original_device)
                safe_print("[VRAM] Memory released, model back on GPU")
            except Exception as e:
                safe_print(f"[VRAM] Cleanup warning: {e}")
        
        self._safe_after(0, lambda: self.btn_stop.config(text="STOP INDEX"))
        self._safe_after(0, lambda: self.progress.configure(value=0))
        self._safe_after(0, lambda: self.progress_label.config(text=""))
        self._safe_after(0, self.update_stats)

        if was_stopped:
            msg = f"Stopped. Saved {count:,} images."
            safe_print(f"[INDEX] {msg}")
            self._safe_after(0, lambda: self.update_status(msg, DANGER))

            if self.pending_action:
                safe_print("[ACTION] Executing pending action...")
                action = self.pending_action
                self.pending_action = None
                self._safe_after(100, action)
            elif count > 0:
                try:
                    query = self.query_entry.get().strip()
                    if query:
                        self._safe_after(500, self.do_search)
                except Exception:
                    pass
        else:
            # Chain video refresh if pending
            if getattr(self, '_pending_video_refresh', False):
                self._pending_video_refresh = False
                if not self.video_cache_file:
                    self.video_cache_file = os.path.join(self.folder, self.get_video_cache_filename())
                self._safe_after(200, lambda: self.start_indexing(mode="video_refresh"))
                return  # messagebox shown after video finishes
            self._safe_after(0, lambda: self.update_status("Indexing Complete", "green"))
            self._safe_after(0, lambda: messagebox.showinfo("Done", f"Index complete.\nTotal images: {count:,}"))
            try:
                query = self.query_entry.get().strip()
                if query:
                    self._safe_after(500, self.do_search)
            except Exception:
                pass

    def _handle_video_stop(self):
        was_stopped = self.stop_indexing
        n_frames = len(self.video_paths)
        n_videos = len(set(vp for vp, _ in self.video_paths)) if self.video_paths else 0

        self.is_indexing = False
        self.stop_indexing = False
        self.is_stopping = False

        if self.clip_model and hasattr(self.clip_model, 'model'):
            import torch
            try:
                if not getattr(self.clip_model, 'onnx_disabled', False):
                    self.clip_model._destroy_onnx_session()
                original_device = self.clip_model.device
                self.clip_model.model.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                gc.collect()
                self.clip_model.model.to(original_device)
            except Exception as e:
                safe_print(f"[VRAM] Video cleanup warning: {e}")

        self._safe_after(0, lambda: self.btn_stop.config(text="STOP INDEX"))
        self._safe_after(0, lambda: self.progress.configure(value=0))
        self._safe_after(0, lambda: self.progress_label.config(text=""))
        self._safe_after(0, self.update_stats)

        if was_stopped:
            msg = f"Stopped. Saved {n_frames:,} frames from {n_videos:,} videos."
            safe_print(f"[VINDEX] {msg}")
            self._safe_after(0, lambda: self.update_status(msg, DANGER))
            if self.pending_action:
                action = self.pending_action
                self.pending_action = None
                self._safe_after(100, action)
        else:
            self._safe_after(0, lambda: self.update_status("Video indexing complete", "green"))
            self._safe_after(0, lambda: messagebox.showinfo(
                "Done", f"Video index complete.\n{n_videos:,} videos | {n_frames:,} frames"
            ))
            try:
                query = self.query_entry.get().strip()
                if query:
                    self._safe_after(500, self.do_search)
            except Exception:
                pass

    def delete_cache(self):
        if not self.folder: return

        # Show exactly what will be destroyed so user knows the stakes
        img_count = len(self.image_paths)
        vid_count = len(set(vp for vp, _ in self.video_paths)) if self.video_paths else 0
        frame_count = len(self.video_paths)

        msg = "⚠ This will permanently DELETE cache files and re-index from scratch.\n\n"
        if img_count:
            msg += f"  • {img_count:,} images will be re-indexed\n"
        if vid_count:
            msg += f"  • {vid_count:,} videos ({frame_count:,} frames) will be re-indexed\n"
        msg += "\nThis cannot be undone. Continue?"

        if not messagebox.askyesno("Delete Cache & Re-Index?", msg, icon="warning"):
            return

        try:
            if self.cache_file and os.path.exists(self.cache_file):
                os.remove(self.cache_file)
                safe_print("[CACHE] Deleted.")
        except: pass

        try:
            if self.video_cache_file and os.path.exists(self.video_cache_file):
                os.remove(self.video_cache_file)
                safe_print("[VCACHE] Deleted.")
        except: pass

        self.image_paths = []
        self.image_embeddings = None
        self.video_paths = []
        self.video_embeddings = None
        self.video_cache_file = None
        self.clear_results()
        self.update_stats()
        self.start_indexing(mode="full")

    def force_reindex(self):
        if not self.folder:
            messagebox.showwarning("Warning", "Select a folder first.")
            return
        # Chain video refresh after image refresh if video index is loaded
        if self.video_paths or self.video_embeddings is not None:
            self._pending_video_refresh = True
        self.start_indexing(mode="refresh")

    def on_index_videos_click(self):
        if not self.is_safe_to_act(action_callback=self.index_videos, action_name="index videos"):
            return
        self.cancel_search(clear_ui=True)
        self.index_videos()

    def index_videos(self):
        if not self.folder:
            messagebox.showwarning("Warning", "Select a folder first.")
            return
        if self.clip_model is None:
            messagebox.showwarning("Wait", "Model is still loading...")
            return
        if not self.video_cache_file:
            self.video_cache_file = os.path.join(self.folder, self.get_video_cache_filename())
        if self.video_paths:
            vid_count = len(set(vp for vp, _ in self.video_paths))
            frame_count = len(self.video_paths)
            answer = messagebox.askyesno(
                "Video Index",
                f"Video index has {vid_count:,} videos ({frame_count:,} frames).\n\n"
                f"Yes = Refresh (add new videos only, keeps existing)\n"
                f"No = Cancel (do nothing)"
            )
            if not answer:
                return  # User pressed No — do nothing, keep existing index safe
            mode = "video_refresh"
        else:
            mode = "video_full"
        self.start_indexing(mode=mode)

    def _deduplicate_video_results(self, all_results):
        """
        For video results, keep only the best scoring frame per video file.
        Image results are kept as-is.
        """
        seen_videos = {}  # abs_video_path -> (score, timestamp)
        deduped = []

        for item in all_results:
            score, path, result_type, metadata = item
            if result_type == "video":
                if path not in seen_videos or score > seen_videos[path][0]:
                    seen_videos[path] = (score, metadata.get("timestamp", 0.0))
            else:
                deduped.append(item)

        # Add best frame for each video
        for path, (score, timestamp) in seen_videos.items():
            deduped.append((score, path, "video", {"timestamp": timestamp}))

        return deduped

    def parse_query(self, query):
        """
        Parse a query string into positive and negative terms.
        Supports: -word or -"multi word phrase" for exclusions.
        Returns (positive_terms, negative_terms) as lists of strings.
        """
        positive_terms = []
        negative_terms = []

        # Match -"quoted phrase", -word, "quoted phrase", or plain word
        pattern = r'(-?"[^"]+"|[-\w]+)'
        tokens = re.findall(pattern, query)

        for token in tokens:
            if token.startswith('-'):
                term = token[1:].strip('"')
                if term:
                    negative_terms.append(term)
            else:
                term = token.strip('"')
                if term:
                    positive_terms.append(term)

        return positive_terms, negative_terms

    def do_search(self):
        if self.is_searching or self.clip_model is None:
            safe_print("[SEARCH] Already searching or model not loaded")
            return
        has_image_data = self.image_embeddings is not None and len(self.image_paths) > 0
        has_video_data = self.video_embeddings is not None and len(self.video_paths) > 0
        if not has_image_data and not has_video_data:
            messagebox.showwarning("No Data", "Index is empty. Please select a folder.")
            return

        query = self.query_entry.get().strip()
        if not query:
            safe_print("[SEARCH] Empty query")
            self.update_status("Enter a search term", "orange")
            return

        safe_print(f"\n[SEARCH] Starting search for: '{query}'")
        self.search_thread = Thread(target=lambda: self.search(query, self.search_generation + 1), daemon=True)
        self.search_thread.start()

    def search(self, query, generation):
        # Only recreate ONNX if it was successfully created before (not permanently disabled)
        if self.clip_model and not getattr(self.clip_model, 'use_onnx_visual', False):
            if not getattr(self.clip_model, 'onnx_disabled', False):
                try:
                    self.clip_model._create_onnx_session()
                except:
                    pass  # Silently fall back to PyTorch
        
        self.search_generation = generation
        self.is_searching = True
        self.stop_search = False  # reset BEFORE anything else to prevent race with thumbnail queue
        self.thumbnail_count = 0  # reset immediately in search thread, don't wait for clear_results
        
        safe_print(f"[SEARCH] Generation: {generation}, Query: '{query}'")
        
        # Clear old results and free RAM before new search
        self.root.after(0, self.clear_results)
        self.total_found = 0
        
        if not self.is_indexing:
            self.root.after(0, lambda: self.update_status("Searching...", "orange"))
            self.root.after(0, lambda: self.progress.config(mode='indeterminate'))
            self.root.after(0, self.progress.start)
            
        try:
            positive_terms, negative_terms = self.parse_query(query)

            if not positive_terms:
                safe_print("[SEARCH] No positive search terms found")
                self.root.after(0, lambda: self.update_status("No positive search terms", "orange"))
                self.is_searching = False
                return

            safe_print(f"[SEARCH] Positive: {positive_terms}, Negative: {negative_terms}")
            safe_print(f"[SEARCH] Encoding query...")

            # Encode positive terms (average if multiple)
            pos_query = " ".join(positive_terms)
            text_embed = self.clip_model.encode_text([pos_query])

            if text_embed is None or text_embed.size == 0:
                safe_print("[SEARCH ERROR] Text encoding returned empty array")
                self.root.after(0, lambda: self.update_status("Search failed - text encoding error", "red"))
                self.is_searching = False
                return

            if self.stop_search or generation != self.search_generation:
                safe_print("[SEARCH] Cancelled after text encoding")
                return

            safe_print(f"[SEARCH] Computing similarities...")

            # Encode negative terms if any
            neg_embed = None
            if negative_terms:
                neg_query = " ".join(negative_terms)
                safe_print(f"[SEARCH] Encoding negative terms: '{neg_query}'")
                neg_embed = self.clip_model.encode_text([neg_query])
                if neg_embed is not None and neg_embed.size > 0:
                    safe_print(f"[SEARCH] Negative terms applied")
                else:
                    neg_embed = None

            if self.stop_search:
                safe_print("[SEARCH] Cancelled after similarity computation")
                return

            if not self.is_indexing:
                self.root.after(0, self.progress.stop)
                self.root.after(0, lambda: self.progress.config(mode='determinate'))

            show_images = self.show_images_var.get() if self.show_images_var else True
            show_videos = self.show_videos_var.get() if self.show_videos_var else True
            all_results = []

            # Flush any pending batches from concurrent indexing before search
            self._flush_pending_batches()

            min_score = self.score_var.get()

            # Search image index
            if show_images and self.image_embeddings is not None and len(self.image_paths) > 0:
                sims_img = (self.image_embeddings @ text_embed.T).flatten()
                if neg_embed is not None:
                    sims_img = sims_img - (self.image_embeddings @ neg_embed.T).flatten()
                # Filter entirely in numpy before building Python list — avoids O(N) Python loop
                # np.where returns only indices above threshold (often <1% of total)
                above = np.where(sims_img >= min_score)[0]
                for i in above:
                    rel_path = self.image_paths[i]
                    if not self._is_excluded(rel_path):
                        abs_path = os.path.join(self.folder, rel_path)
                        all_results.append((float(sims_img[i]), abs_path, "image", {}))

            # Search video index
            if show_videos and self.video_embeddings is not None and len(self.video_paths) > 0:
                sims_vid = (self.video_embeddings @ text_embed.T).flatten()
                if neg_embed is not None:
                    sims_vid = sims_vid - (self.video_embeddings @ neg_embed.T).flatten()
                above_v = np.where(sims_vid >= min_score)[0]
                for i in above_v:
                    rel_vid_path, timestamp = self.video_paths[i]
                    if not self._is_excluded(rel_vid_path):
                        abs_vid_path = os.path.join(self.folder, rel_vid_path)
                        all_results.append((float(sims_vid[i]), abs_vid_path, "video", {"timestamp": timestamp}))

            safe_print(f"[SEARCH] Found {len(all_results)} total results")

            # Deduplicate video results — only if toggle is on
            if self.dedup_video_var.get():
                all_results = self._deduplicate_video_results(all_results)

            # Sort ALL results by score, store in memory
            all_results.sort(key=lambda x: x[0], reverse=True)
            # (score threshold already applied via numpy np.where above — no second filter needed)
            self.all_search_results = all_results
            self.total_found = len(all_results)
            self.show_more_offset = 0

            if all_results:
                # Page 1 — cap at 100 to stay under Tkinter's 32,767px canvas limit
                initial_n = max(10, self.top_n_var.get())
                first_batch = all_results[:initial_n]
                self.show_more_offset = len(first_batch)

                safe_print(f"[SEARCH] Displaying first {len(first_batch)} of {self.total_found} results")

                # Suggest lowering score if very few results
                if self.total_found < 6 and not self.is_indexing:
                    self.root.after(500, self._maybe_suggest_lower_score)

                cw = max(self.canvas.winfo_width(), CELL_WIDTH)
                self.render_cols = max(1, cw // CELL_WIDTH)

                self.start_thumbnail_loader(first_batch, generation)
            else:
                safe_print("[SEARCH] No results found")
                if not self.is_indexing:
                    self.root.after(0, lambda: self.update_status("No results found", "green"))
                    self.root.after(100, self._maybe_suggest_lower_score)
                self.is_searching = False
                
        except Exception as e:
            if not self.stop_search:
                safe_print(f"[SEARCH ERROR] {e}")
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: self.update_status("Search error - check console", "red"))
            self.is_searching = False

    def image_search(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.webp")])
        if not path: return
        gen = self.search_generation + 1
        self.search_thread = Thread(target=lambda: self._image_search(path, gen), daemon=True)
        self.search_thread.start()

    def _image_search(self, path, generation):
        # Only recreate ONNX if it's not permanently disabled
        if self.clip_model and not getattr(self.clip_model, 'use_onnx_visual', False):
            if not getattr(self.clip_model, 'onnx_disabled', False):
                try:
                    self.clip_model._create_onnx_session()
                except:
                    pass  # Silently fall back to PyTorch
        
        self.search_generation = generation
        self.is_searching = True
        self.stop_search = False  # reset BEFORE anything else to prevent race with thumbnail queue
        self.thumbnail_count = 0  # reset immediately in search thread
        
        # Clear old results and free RAM before new search
        self.root.after(0, self.clear_results)
        
        cw = max(self.canvas.winfo_width(), CELL_WIDTH)
        self.render_cols = max(1, cw // CELL_WIDTH)
        
        self.root.after(0, lambda: self.update_status("Searching by image...", "orange"))
        
        try:
            img = open_image(path)
            if img is None:
                self.is_searching = False
                return
            features = self.clip_model.encode_image_batch([img])
            emb = features[0]
            
            show_images = self.show_images_var.get() if self.show_images_var else True
            show_videos = self.show_videos_var.get() if self.show_videos_var else True
            all_results = []
            min_score = self.score_var.get()

            if show_images and self.image_embeddings is not None:
                sims_img = (self.image_embeddings @ emb).flatten()
                above = np.where(sims_img >= min_score)[0]
                for i in above:
                    rel_path = self.image_paths[i]
                    if not self._is_excluded(rel_path):
                        abs_path = os.path.join(self.folder, rel_path)
                        all_results.append((float(sims_img[i]), abs_path, "image", {}))

            if show_videos and self.video_embeddings is not None:
                sims_vid = (self.video_embeddings @ emb).flatten()
                above_v = np.where(sims_vid >= min_score)[0]
                for i in above_v:
                    rel_vid_path, timestamp = self.video_paths[i]
                    if not self._is_excluded(rel_vid_path):
                        abs_vid_path = os.path.join(self.folder, rel_vid_path)
                        all_results.append((float(sims_vid[i]), abs_vid_path, "video", {"timestamp": timestamp}))

            if all_results:
                if self.dedup_video_var.get():
                    all_results = self._deduplicate_video_results(all_results)
                all_results.sort(key=lambda x: x[0], reverse=True)
                # (score threshold already applied via numpy np.where above)
                self.all_search_results = all_results
                self.total_found = len(all_results)
                self.show_more_offset = 0

                initial_n = max(10, self.top_n_var.get())
                first_batch = all_results[:initial_n]
                self.show_more_offset = len(first_batch)

                self.start_thumbnail_loader(first_batch, generation)
            else:
                self.root.after(0, lambda: self.update_status("No matches", "green"))
                self.is_searching = False
        except Exception as e:
            safe_print(f"[IMAGE SEARCH ERROR] {e}")
            self.is_searching = False

    def start_thumbnail_loader(self, results, generation):
        safe_print(f"[THUMBNAILS] Starting loader for {len(results)} results")
        with self.thumbnail_queue.mutex:
            self.thumbnail_queue.queue.clear()
        t = Thread(target=self.load_thumbnails_worker, args=(results, generation), daemon=True)
        self._thumbnail_worker_thread = t
        t.start()
        self.root.after(10, lambda: self.check_thumbnail_queue(generation))

    def load_thumbnails_worker(self, results, generation):
        loaded = 0
        failed = 0
        for item in results:
            score, path, result_type, metadata = item
            if self.stop_search or generation != self.search_generation:
                safe_print(f"[THUMBNAILS] Stopped (loaded {loaded}, failed {failed})")
                break
            try:
                if result_type == "image":
                    if path.lower().endswith(RAW_EXTS):
                        # RAW files need rawpy — PIL cannot decode them natively
                        try:
                            import rawpy
                            with rawpy.imread(path) as raw:
                                rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)
                            img = Image.fromarray(rgb)
                        except ImportError:
                            failed += 1
                            continue
                        except Exception:
                            failed += 1
                            continue
                    else:
                        safe_path = get_safe_path(path)
                        with open(safe_path, 'rb') as fh:
                            img = Image.open(fh)
                            if img.mode == 'P' and 'transparency' in img.info:
                                img = img.convert("RGBA")
                            img.load()
                    img.thumbnail(THUMBNAIL_SIZE)
                elif result_type == "video":
                    try:
                        import cv2
                    except ImportError:
                        if not getattr(self, '_cv2_missing_warned', False):
                            self._cv2_missing_warned = True
                            self.root.after(0, lambda: messagebox.showwarning(
                                "Missing Dependency",
                                "OpenCV is not installed - video thumbnails cannot be displayed.\n\n"
                                "Install it with:\n    pip install opencv-python"
                            ))
                        failed += 1
                        continue
                    timestamp = metadata.get("timestamp", 0.0)
                    cap = cv2.VideoCapture(path)
                    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                    ret, frame = cap.read()
                    cap.release()
                    if not ret or frame is None:
                        failed += 1
                        continue
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    img.thumbnail(THUMBNAIL_SIZE)
                else:
                    failed += 1
                    continue
                self.thumbnail_queue.put((score, path, img, result_type, metadata))
                loaded += 1
            except Exception as e:
                failed += 1
        safe_print(f"[THUMBNAILS] Completed: {loaded} loaded, {failed} failed")

    def check_thumbnail_queue(self, generation):
        if self.stop_search or generation != self.search_generation: 
            safe_print(f"[THUMBNAILS] Queue check stopped")
            return
        
        start_time = time.time()
        processed_this_cycle = 0
        while not self.thumbnail_queue.empty():
            try:
                item = self.thumbnail_queue.get_nowait()
                score, path, img, result_type, metadata = item
                self.add_result_thumbnail(score, path, img, result_type, metadata)
                processed_this_cycle += 1
            except queue.Empty: break
            
            if time.time() - start_time > 0.1: break
        
        # Update scrollregion once per cycle, not once per thumbnail
        if processed_this_cycle > 0:
            self.results_frame.update_idletasks()
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        done = self.thumbnail_count
        
        if processed_this_cycle > 0:
            safe_print(f"[THUMBNAILS] Displayed {done} results", end='\r')
        
        if not self.is_indexing:
            page_size = max(10, self.top_n_var.get())
            total_pages = max(1, -(-self.total_found // page_size)) if self.total_found > 0 else 1
            # current_page = which page is currently DISPLAYED (offset already advanced to next page)
            # so displayed page = (offset - page_size) // page_size + 1 = offset // page_size
            current_page = max(1, (self.show_more_offset - 1) // page_size + 1) if self.show_more_offset > 0 else 1
            self.progress_label.config(text=f"Page {current_page} of {total_pages}  —  {self.total_found:,} total results")

        # Queue is empty — check if all worker threads are done
        if not self.thumbnail_queue.empty():
            self.root.after(10, lambda: self.check_thumbnail_queue(generation))
            return

        # Give worker thread a tiny moment to push final items then re-check once
        active_thread = getattr(self, '_thumbnail_worker_thread', None)
        if active_thread and active_thread.is_alive():
            self.root.after(20, lambda: self.check_thumbnail_queue(generation))
            return

        # All done
        self.is_searching = False
        if not self.is_indexing:
            safe_print(f"\n[THUMBNAILS] Display complete: {done} shown of {self.total_found:,} total")
            self.update_status(f"Found {self.total_found:,} results", "green")
        # Ensure scrollregion recalculated after last thumbnails added
        self.results_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.root.after(0, self._update_show_more_button)

    def _update_show_more_button(self):
        """Update page navigation buttons and page label"""
        page_size = max(10, self.top_n_var.get())  # use slider value directly, no cap
        total = len(self.all_search_results)
        if total == 0:
            self.show_more_frame.pack_forget()
            return

        current_page = max(1, (self.show_more_offset - 1) // page_size + 1) if self.show_more_offset > 0 else 1
        total_pages = max(1, -(-total // page_size))  # ceiling division

        self.page_label.config(text=f"Page {current_page} of {total_pages}")

        # Prev button — only if not on first page
        self.prev_page_btn.config(state="normal" if self.show_more_offset > page_size else "disabled")

        # Next button — only if more results remain
        self.show_more_btn.config(
            state="normal" if self.show_more_offset < total else "disabled",
            text="Next Page →"
        )

        self.show_more_frame.pack(fill="x", padx=8, pady=(0, 6))

    def prev_page_results(self):
        """Go back one page"""
        if self.is_searching or not self.all_search_results:
            return
        page_size = max(10, self.top_n_var.get())
        # Go back two page_sizes from current offset (one for current page, one for prev)
        new_offset = max(0, self.show_more_offset - (page_size * 2))
        prev_batch = self.all_search_results[new_offset:new_offset + page_size]
        if not prev_batch:
            return

        saved_results = self.all_search_results
        saved_total = self.total_found

        self.clear_results(keep_results=True)

        self.all_search_results = saved_results
        self.total_found = saved_total
        self.show_more_offset = new_offset + len(prev_batch)

        self.stop_search = False
        gen = self.search_generation
        t = Thread(target=self.load_thumbnails_worker, args=(prev_batch, gen), daemon=True)
        self._thumbnail_worker_thread = t
        t.start()
        self.root.after(10, lambda: self.check_thumbnail_queue(gen))

    def show_more_results(self):
        """Clear current page widgets and load next results"""
        if self.is_searching or not self.all_search_results:
            return
        page_size = max(10, self.top_n_var.get())
        next_batch = self.all_search_results[self.show_more_offset:self.show_more_offset + page_size]
        if not next_batch:
            return

        # Remember state before clearing
        saved_results = self.all_search_results
        saved_total = self.total_found
        new_offset = self.show_more_offset + len(next_batch)

        # Clear widgets only — keep_results=True so all_search_results is NOT wiped
        self.clear_results(keep_results=True)

        # Restore state that clear_results didn't touch (thumbnail_count reset by clear_results)
        self.all_search_results = saved_results
        self.total_found = saved_total
        self.show_more_offset = new_offset

        self.stop_search = False
        gen = self.search_generation
        t = Thread(target=self.load_thumbnails_worker, args=(next_batch, gen), daemon=True)
        self._thumbnail_worker_thread = t
        t.start()
        self.root.after(10, lambda: self.check_thumbnail_queue(gen))

    def add_result_thumbnail(self, score, path, pil_img, result_type="image", metadata=None):
        if self.stop_search: return
        if metadata is None:
            metadata = {}

        # Unique cache key — video frames need per-frame uniqueness
        if result_type == "video":
            ts = metadata.get("timestamp", 0.0)
            cache_key = f"{path}@{ts}"
        else:
            cache_key = path

        cols = max(1, getattr(self, "render_cols", 1))
        idx = self.thumbnail_count
        row, col = divmod(idx, cols)
        self.thumbnail_count += 1

        f = tk.Frame(self.results_frame, bg=CARD_BG, bd=1, relief="solid")
        f.grid(row=row, column=col, padx=6, pady=6)
        f.configure(width=CELL_WIDTH, height=CELL_HEIGHT)
        f.pack_propagate(False)
        f._image_path = path  # always the file path (not timestamp) for Copy/Move/Delete

        try:
            photo = ImageTk.PhotoImage(pil_img)

            # Prune oldest thumbnails if cache exceeds limit
            if len(self.thumbnail_images) >= MAX_THUMBNAIL_CACHE:
                oldest_keys = list(self.thumbnail_images.keys())[:len(self.thumbnail_images) - MAX_THUMBNAIL_CACHE + 1]
                for k in oldest_keys:
                    del self.thumbnail_images[k]

            self.thumbnail_images[cache_key] = photo

            # VIDEO badge
            if result_type == "video":
                tk.Label(f, text="VIDEO", bg=ORANGE, fg="#000000",
                         font=("Segoe UI", 7, "bold"), padx=3, pady=1).pack(anchor="nw", padx=4, pady=(4, 0))

            lbl = tk.Label(f, image=photo, bg=CARD_BG)
            lbl.image = photo  # keep reference alive tied to widget, prevents GC blank images
            lbl.pack(pady=4)

            lbl.bind("<Button-1>", lambda e: self.handle_single_click(path))
            lbl.bind("<Double-Button-1>", lambda e: self.handle_double_click(path))
            lbl.bind("<Button-3>", lambda e, p=path: self._show_card_context_menu(e, p))
            f.bind("<Button-3>", lambda e, p=path: self._show_card_context_menu(e, p))

            var = tk.BooleanVar(value=(path in self.selected_images))
            cb = tk.Checkbutton(f, text="Select", variable=var, bg=CARD_BG, fg=FG,
                                selectcolor=BG, command=lambda p=path, v=var: self.toggle_selection(p, v.get()))
            cb.var = var   # keep reference alive — GC would destroy a local BooleanVar
            cb.pack()

            name = os.path.basename(path)
            if len(name) > 25: name = name[:22] + "..."

            if result_type == "video":
                ts = metadata.get("timestamp", 0.0)
                minutes = int(ts) // 60
                seconds = int(ts) % 60
                label_text = f"{score:.3f}\n{name}\nt={minutes}:{seconds:02d}"
            else:
                label_text = f"{score:.3f}\n{name}"

            tk.Label(f, text=label_text, bg=CARD_BG, fg=FG,
                     font=("Segoe UI", 9), wraplength=180, justify="center").pack(pady=2)
        except:
            f.destroy()
            self.thumbnail_count -= 1  # undo the increment so grid has no gaps

    def handle_single_click(self, path):
        if self.click_timer:
            self.root.after_cancel(self.click_timer)
        self.click_timer = self.root.after(250, lambda: self.open_in_explorer(path))

    def handle_double_click(self, path):
        if self.click_timer:
            self.root.after_cancel(self.click_timer)
            self.click_timer = None
        self.open_image_viewer(path)

    def open_in_explorer(self, path):
        """Open file location - path is already absolute from search results"""
        self.click_timer = None
        if os.path.exists(path):
            path = os.path.normpath(path)
            if os.name == 'nt':
                # Strip \\?\ prefix — Windows Explorer CLI rejects extended path prefixes
                explorer_path = path.replace('\\\\?\\', '').replace('\\??\\', '')
                subprocess.Popen(f'explorer /select,"{explorer_path}"')
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', '-R', path])
            else:
                folder = os.path.dirname(path)
                subprocess.Popen(['xdg-open', folder])

    def open_image_viewer(self, path):
        """Open image - path is already absolute from search results"""
        if os.path.exists(path):
            if os.name == 'nt':
                os.startfile(path)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', path])
            else:
                subprocess.Popen(['xdg-open', path])

    def clear_results(self, keep_results=False):
        """Clear search results and free RAM from thumbnails"""
        
        # Destroy all thumbnail widgets
        for w in self.results_frame.winfo_children():
            w.destroy()
        
        self.thumbnail_count = 0
        self.show_more_frame.pack_forget()
        
        if not keep_results:
            # Full clear — wipe thumbnail cache and all result state
            # On pagination (keep_results=True) we keep cache so page turns
            # don't re-decode every image from disk unnecessarily
            self.thumbnail_images.clear()
            self.all_search_results = []
            self.show_more_offset = 0
            self.total_found = 0
            self.selected_images.clear()  # clear selections on new search, not on page turns

        # Force garbage collection to free RAM immediately
        gc.collect()
        
        self.canvas.yview_moveto(0)


    def _maybe_suggest_lower_score(self):
        """Show a dismissable hint when search returns very few or no results"""
        current_score = self.score_var.get()
        if current_score > 0.05:
            messagebox.showinfo(
                "Few Results Found",
                f"Only {self.total_found} result(s) found at similarity score {current_score:.2f}.\n\n"
                f"Try lowering the Similarity Score slider to find more matches.\n"
                f"• Text search works well at 0.15–0.30\n"
                f"• Image search works well at 0.60–0.85"
            )

    def _get_all_cards(self):
        """Return all card frames currently displayed in results_frame"""
        return [w for w in self.results_frame.winfo_children()
                if isinstance(w, tk.Frame)]

    def _select_card(self, card, select=True):
        """Programmatically select or deselect a card and update its checkbox"""
        path = getattr(card, '_image_path', None)
        if path is None:
            return
        if select:
            self.selected_images.add(path)
        else:
            self.selected_images.discard(path)
        for child in card.winfo_children():
            if isinstance(child, tk.Checkbutton):
                try:
                    if hasattr(child, 'var'):
                        child.var.set(select)  # BooleanVar drives visual state
                    elif select:
                        child.select()
                    else:
                        child.deselect()
                except Exception:
                    pass
                break

    def _setup_rubber_band(self):
        """Windows-Explorer-style rubber-band selection.

        Design:
        - Press anywhere inside the canvas area → enters PENDING state
        - Move >5px → enters ACTIVE state, draws selection rectangle
        - Release with <5px total movement → normal click, card handlers fire untouched
        - Release with >5px movement → selects all cards inside rectangle, cancels any click timer

        Why bind_all for motion/release:
        Tkinter's implicit pointer grab sends all B1-Motion and ButtonRelease-1 events to
        whichever widget received the original ButtonPress-1, which is often a card's child
        label or checkbox — not results_frame.  bind_all catches them globally; the
        _rb_pending guard ensures we only act when we started from within the canvas area.

        Why bind_all for press too:
        results_frame.bind("<ButtonPress-1>") only fires when clicking on the frame's own
        background pixels, not on child card widgets that cover it.  bind_all + canvas-
        bounds check lets the drag start from anywhere inside the scrollable grid.

        Overlay Toplevel:
        Tkinter canvas create_rectangle items are always rendered BEHIND embedded create_window
        widgets (the card frames), making the selection box invisible. Instead we use a
        transparent Toplevel that floats above everything and is withdrawn on release.
        """
        self._rb_start_x = 0
        self._rb_start_y = 0
        self._rb_rect = None
        self._rb_active = False   # True once mouse moved >5px — rectangle is visible
        self._rb_pending = False  # True from press until release — waiting for 5px threshold

        # --- Overlay Toplevel for visible rubber-band rectangle ---
        # canvas.create_rectangle() is always hidden behind card widgets (Tkinter limitation)
        # A transparent Toplevel placed over the canvas fixes this.
        self._rb_overlay = None
        self._rb_overlay_canvas = None
        self._rb_overlay_rect = None
        try:
            overlay = tk.Toplevel(self.root)
            overlay.overrideredirect(True)   # no title bar / borders
            overlay.withdraw()               # hidden until drag starts
            if os.name == 'nt':
                # Windows: declare a near-black colour as transparent so the background
                # disappears, leaving only the drawn rectangle visible
                _TRANS = '#000002'
                overlay.attributes('-transparentcolor', _TRANS)
                ovcanvas = tk.Canvas(overlay, bg=_TRANS, highlightthickness=0)
            else:
                # macOS / Linux fallback: very high alpha so the BG is nearly invisible
                overlay.attributes('-alpha', 0.01)
                ovcanvas = tk.Canvas(overlay, bg='black', highlightthickness=0)
            ovcanvas.pack(fill='both', expand=True)
            self._rb_overlay = overlay
            self._rb_overlay_canvas = ovcanvas
        except Exception:
            # If overlay creation fails for any reason, fall back to canvas drawing
            self._rb_overlay = None
            self._rb_overlay_canvas = None

        # bind_all so events reach us regardless of which child widget has the pointer grab
        self.root.bind_all("<ButtonPress-1>",   self._rb_on_press)
        self.root.bind_all("<B1-Motion>",       self._rb_on_drag)
        self.root.bind_all("<ButtonRelease-1>", self._rb_on_release)

        # Right-click context menu on empty canvas background (unchanged)
        self.canvas.bind("<ButtonPress-3>", self._show_canvas_context_menu)

    def _rb_screen_to_canvas(self, x_root, y_root):
        """Convert screen coordinates to canvas scroll-adjusted coordinates."""
        local_x = x_root - self.canvas.winfo_rootx()
        local_y = y_root - self.canvas.winfo_rooty()
        return self.canvas.canvasx(local_x), self.canvas.canvasy(local_y)

    def _rb_on_press(self, event):
        """Start tracking a potential rubber-band drag.

        Only activates if the press lands inside the canvas widget area.
        All other clicks (toolbar buttons, search box, etc.) are ignored.
        """
        # Check whether the click is within the canvas bounds
        cx = self.canvas.winfo_rootx()
        cy = self.canvas.winfo_rooty()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if not (cx <= event.x_root <= cx + cw and cy <= event.y_root <= cy + ch):
            return  # Click is outside canvas area — ignore completely

        self._rb_pending = True
        self._rb_active = False
        self._rb_start_x, self._rb_start_y = self._rb_screen_to_canvas(event.x_root, event.y_root)
        if self._rb_rect:
            self.canvas.delete(self._rb_rect)
            self._rb_rect = None

    def _rb_on_drag(self, event):
        """Draw the selection rectangle once the drag threshold is crossed."""
        if not self._rb_pending:
            return

        cx, cy = self._rb_screen_to_canvas(event.x_root, event.y_root)
        dx = abs(cx - self._rb_start_x)
        dy = abs(cy - self._rb_start_y)

        # Only commit to rubber-band after 5px movement — below that it's still a click
        if not self._rb_active:
            if dx < 5 and dy < 5:
                return
            self._rb_active = True
            # Cancel any card single-click timer so drag doesn't trigger open-in-explorer
            if self.click_timer:
                self.root.after_cancel(self.click_timer)
                self.click_timer = None
            # Show and position overlay ONCE when drag activates (not every motion event)
            # Calling geometry() + deiconify() on every pixel of movement causes flickering
            if self._rb_overlay and self._rb_overlay_canvas:
                canvas_x = self.canvas.winfo_rootx()
                canvas_y = self.canvas.winfo_rooty()
                canvas_w = self.canvas.winfo_width()
                canvas_h = self.canvas.winfo_height()
                self._rb_overlay.geometry(f'{canvas_w}x{canvas_h}+{canvas_x}+{canvas_y}')
                self._rb_overlay.deiconify()
                self._rb_overlay.lift()

        # Clean up old canvas rect if any (fallback path)
        if self._rb_rect:
            self.canvas.delete(self._rb_rect)
            self._rb_rect = None

        if self._rb_overlay and self._rb_overlay_canvas:
            # --- Overlay path (Windows / most platforms) ---
            # Only update the rectangle — overlay position was set when drag first activated

            # Convert canvas-scroll coordinates to overlay-local pixel coordinates
            # (overlay top-left = canvas widget's top-left on screen)
            scroll_x0 = self.canvas.canvasx(0)
            scroll_y0 = self.canvas.canvasy(0)
            r_x1 = self._rb_start_x - scroll_x0
            r_y1 = self._rb_start_y - scroll_y0
            r_x2 = cx - scroll_x0
            r_y2 = cy - scroll_y0

            if self._rb_overlay_rect:
                self._rb_overlay_canvas.delete(self._rb_overlay_rect)
            self._rb_overlay_rect = self._rb_overlay_canvas.create_rectangle(
                r_x1, r_y1, r_x2, r_y2,
                outline="#4CAF50", fill="", width=2, dash=(4, 2)
            )
        else:
            # --- Fallback: draw on main canvas (rect may be hidden behind cards) ---
            if self._rb_rect:
                self.canvas.delete(self._rb_rect)
            # fill="" = transparent interior (Tkinter does not support 8-digit hex alpha colours)
            self._rb_rect = self.canvas.create_rectangle(
                self._rb_start_x, self._rb_start_y, cx, cy,
                outline="#4CAF50", fill="", width=2, dash=(4, 2)
            )

    def _rb_on_release(self, event):
        """On release: if rubber-band was active select cards; otherwise let normal click fire."""
        if not self._rb_pending:
            return

        was_active = self._rb_active
        self._rb_pending = False
        self._rb_active = False

        # Clean up canvas rect (fallback path)
        if self._rb_rect:
            self.canvas.delete(self._rb_rect)
            self._rb_rect = None

        # Clean up overlay rect and hide the Toplevel
        if self._rb_overlay and self._rb_overlay_canvas:
            if self._rb_overlay_rect:
                self._rb_overlay_canvas.delete(self._rb_overlay_rect)
                self._rb_overlay_rect = None
            self._rb_overlay.withdraw()

        if not was_active:
            # Drag was under 5px — treat as a normal click, do not interfere
            return

        cx, cy = self._rb_screen_to_canvas(event.x_root, event.y_root)
        x1 = min(self._rb_start_x, cx)
        y1 = min(self._rb_start_y, cy)
        x2 = max(self._rb_start_x, cx)
        y2 = max(self._rb_start_y, cy)

        if abs(x2 - x1) < 5 and abs(y2 - y1) < 5:
            return

        deselect_mode = bool(event.state & 0x4)  # Ctrl held = deselect mode

        # results_frame origin in canvas-scroll coordinates
        frame_canvas_x = (self.results_frame.winfo_rootx() - self.canvas.winfo_rootx()
                          + self.canvas.canvasx(0))
        frame_canvas_y = (self.results_frame.winfo_rooty() - self.canvas.winfo_rooty()
                          + self.canvas.canvasy(0))

        for card in self._get_all_cards():
            try:
                # card.winfo_x/y is relative to results_frame → add frame offset
                card_x = card.winfo_x() + frame_canvas_x
                card_y = card.winfo_y() + frame_canvas_y
                card_w = card.winfo_width()
                card_h = card.winfo_height()
                if card_x < x2 and card_x + card_w > x1 and card_y < y2 and card_y + card_h > y1:
                    self._select_card(card, select=not deselect_mode)
            except Exception:
                pass

    def _show_canvas_context_menu(self, event):
        """Right-click on canvas background — general context menu"""
        menu = tk.Menu(self.root, tearoff=0, bg=CARD_BG, fg=FG,
                       activebackground=ACCENT, activeforeground="#ffffff",
                       relief="flat", bd=1)
        menu.add_command(label="Select All", command=self._select_all_cards)
        menu.add_command(label="Deselect All", command=self._deselect_all_cards)
        menu.add_separator()
        menu.add_command(label="Copy Selected", command=self.export_selected)
        menu.add_command(label="Move Selected", command=self.move_selected)
        menu.add_command(label="Delete Selected", command=self.delete_selected)
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _show_card_context_menu(self, event, path):
        """Right-click on a specific card"""
        menu = tk.Menu(self.root, tearoff=0, bg=CARD_BG, fg=FG,
                       activebackground=ACCENT, activeforeground="#ffffff",
                       relief="flat", bd=1)
        is_selected = path in self.selected_images
        if is_selected:
            menu.add_command(label="Deselect",
                command=lambda: self._set_card_selection_by_path(path, False))
        else:
            menu.add_command(label="Select",
                command=lambda: self._set_card_selection_by_path(path, True))
        menu.add_separator()
        menu.add_command(label="Copy", command=self.export_selected)
        menu.add_command(label="Move", command=self.move_selected)
        menu.add_command(label="Delete", command=self.delete_selected)
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _set_card_selection_by_path(self, path, select):
        """Select or deselect all cards matching path — no break so all video timestamp
        thumbnails of the same file get their checkmark updated, not just the first one."""
        for card in self._get_all_cards():
            if getattr(card, '_image_path', None) == path:
                self._select_card(card, select=select)

    def _select_all_cards(self):
        for card in self._get_all_cards():
            self._select_card(card, select=True)

    def _deselect_all_cards(self):
        for card in self._get_all_cards():
            self._select_card(card, select=False)

    def toggle_selection(self, path, selected):
        if selected: 
            self.selected_images.add(path)
        else: 
            self.selected_images.discard(path)

    def export_selected(self):
        if not self.selected_images:
            messagebox.showinfo("Info", "No images selected")
            return
        export_dir = filedialog.askdirectory(title="Export to")
        if not export_dir: return
        copied = 0
        skipped = 0   # destination file already exists
        errors = []
        for path in self.selected_images:
            try:
                dest = os.path.join(export_dir, os.path.basename(path))
                if os.path.exists(dest):
                    skipped += 1
                    continue
                shutil.copy2(path, export_dir)
                copied += 1
            except Exception as e:
                errors.append(f"{os.path.basename(path)}: {e}")
        # Build a clear summary message
        lines = []
        if copied:
            lines.append(f"✓  {copied} file(s) copied successfully")
        if skipped:
            lines.append(f"⚠  {skipped} file(s) skipped — already exist at destination")
        if errors:
            lines.append(f"✗  {len(errors)} file(s) failed:")
            lines.extend(f"    {e}" for e in errors[:5])
            if len(errors) > 5:
                lines.append(f"    … and {len(errors) - 5} more")
        messagebox.showinfo("Copy Complete", "\n".join(lines) if lines else "Nothing to copy.")
        self.selected_images.clear()

    def _remove_paths_from_index(self, abs_paths):
        if not self.folder:
            return
        rel_to_remove = set()
        for p in abs_paths:
            try:
                rel_to_remove.add(os.path.relpath(p, self.folder))
            except ValueError:
                pass  # different drive (Windows edge case)

        if not rel_to_remove:
            return

        keep_indices = [i for i, rp in enumerate(self.image_paths) if rp not in rel_to_remove]
        self.image_paths = [self.image_paths[i] for i in keep_indices]
        if self.image_embeddings is not None:
            if keep_indices:
                self.image_embeddings = self.image_embeddings[keep_indices]
            else:
                self.image_embeddings = None
        self._save_cache(allow_shrink=True)

        # Also prune video frames for any deleted video files
        if self.video_paths and rel_to_remove:
            keep_video = [i for i, (vp, _) in enumerate(self.video_paths) if vp not in rel_to_remove]
            if len(keep_video) < len(self.video_paths):
                if self.video_embeddings is not None:
                    self.video_embeddings = self.video_embeddings[keep_video] if keep_video else None
                self.video_paths = [self.video_paths[i] for i in keep_video]
                self._save_video_cache(allow_shrink=True)

        self.update_stats()

    def _remove_cards_from_ui(self, abs_paths):
        paths_set = set(abs_paths)
        cards_to_destroy = []
        for card in list(self.results_frame.winfo_children()):
            card_path = getattr(card, '_image_path', None)
            if card_path in paths_set:
                cards_to_destroy.append(card)
                if card_path in self.thumbnail_images:
                    del self.thumbnail_images[card_path]

        for card in cards_to_destroy:
            card.destroy()

        # Re-grid remaining cards to fill gaps
        cols = max(1, getattr(self, 'render_cols', 1))
        for idx, card in enumerate(list(self.results_frame.winfo_children())):
            row, col = divmod(idx, cols)
            card.grid(row=row, column=col, padx=6, pady=6)

    def move_selected(self):
        if not self.selected_images:
            messagebox.showinfo("Info", "No images selected")
            return
        dest_dir = filedialog.askdirectory(title="Move selected images to...")
        if not dest_dir:
            return
        moved = []
        skipped_paths = []  # for UI removal: actually skipped means NOT moved
        skipped = 0
        errors = []
        for path in list(self.selected_images):
            try:
                clean_path = path.replace('\\\\?\\', '').replace('\\??\\', '')
                clean_path = os.path.normpath(os.path.abspath(clean_path))
                dest = os.path.join(dest_dir, os.path.basename(clean_path))
                if os.path.exists(dest):
                    skipped += 1
                    continue
                shutil.move(clean_path, dest_dir)
                moved.append(path)
            except Exception as e:
                errors.append(f"{os.path.basename(path)}: {e}")
        if moved:
            self._remove_cards_from_ui(moved)
            self._remove_paths_from_index(moved)
        self.selected_images.clear()
        # Build a clear summary message
        lines = []
        if moved:
            lines.append(f"✓  {len(moved)} file(s) moved successfully")
        if skipped:
            lines.append(f"⚠  {skipped} file(s) skipped — already exist at destination")
        if errors:
            lines.append(f"✗  {len(errors)} file(s) failed:")
            lines.extend(f"    {e}" for e in errors[:5])
            if len(errors) > 5:
                lines.append(f"    … and {len(errors) - 5} more")
        messagebox.showinfo("Move Complete", "\n".join(lines) if lines else "Nothing to move.")
        self.update_status(f"Moved {len(moved)} images", "green")

    def delete_selected(self):
        if not self.selected_images:
            messagebox.showinfo("Info", "No images selected")
            return

        # Check send2trash BEFORE asking user — never fall back to permanent delete
        try:
            from send2trash import send2trash
        except ImportError:
            messagebox.showerror(
                "Missing Dependency",
                "Cannot delete safely — 'send2trash' is not installed.\n\n"
                "Files will NOT be deleted. Install it with:\n"
                "    pip install send2trash\n\n"
                "This ensures files go to the Recycle Bin and can be recovered."
            )
            return

        count = len(self.selected_images)
        if not messagebox.askyesno(
            "Confirm Delete",
            f"Move {count} selected file(s) to the Recycle Bin?\nFiles can be restored from the Recycle Bin."
        ):
            return

        deleted = []
        errors = []
        for path in list(self.selected_images):
            # Normalize path — strip any \\?\ prefix, convert slashes, resolve to absolute
            clean_path = path.replace('\\\\?\\', '').replace('\\??\\', '')
            clean_path = os.path.normpath(os.path.abspath(clean_path))
            safe_print(f"[DELETE] Attempting: {repr(clean_path)}, exists={os.path.exists(clean_path)}")
            try:
                send2trash(clean_path)
                deleted.append(path)
            except Exception as e:
                errors.append(f"{os.path.basename(path)}: {e}")

        if deleted:
            self._remove_cards_from_ui(deleted)
            self._remove_paths_from_index(deleted)
        self.selected_images.clear()

        # Separate success and error messages — never mix them in one dialog
        if deleted:
            messagebox.showinfo("Moved to Recycle Bin",
                f"Successfully moved {len(deleted)} file(s) to the Recycle Bin.")
            self.update_status(f"Moved {len(deleted)} files to Recycle Bin", "green")
        if errors:
            messagebox.showerror("Delete Errors",
                f"{len(errors)} file(s) could not be deleted:\n\n" + "\n".join(errors[:8]))

    def update_status(self, text, color="blue"):
        self.status_label.config(text=text, foreground=color)

    def update_stats(self):
        has_images = self.image_embeddings is not None and len(self.image_paths) > 0
        has_videos = self.video_embeddings is not None and len(self.video_paths) > 0

        if has_images and has_videos:
            n_imgs = len(self.image_paths)
            n_frames = len(self.video_paths)
            n_vids = len(set(vp for vp, _ in self.video_paths))
            self.stats_label.config(text=f"{n_imgs:,} images | {n_vids:,} videos ({n_frames:,} frames)")
        elif has_images:
            self.stats_label.config(text=f"{len(self.image_paths):,} images indexed")
        elif has_videos:
            n_frames = len(self.video_paths)
            n_vids = len(set(vp for vp, _ in self.video_paths))
            self.stats_label.config(text=f"{n_vids:,} videos ({n_frames:,} frames)")
        else:
            self.stats_label.config(text="")

    def update_progress(self, value, text):
        self.progress['value'] = value
        self.progress_label.config(text=text)

    def show_index_info(self):
        folder_str = self.folder if self.folder else "No folder selected"
        cache_str = self.cache_file if self.cache_file else "No cache loaded"

        cache_size_str = "N/A"
        cache_mtime_str = "N/A"
        if self.cache_file and os.path.exists(self.cache_file):
            try:
                size_bytes = os.path.getsize(self.cache_file)
                if size_bytes >= 1024 ** 3:
                    cache_size_str = f"{size_bytes / 1024**3:.2f} GB"
                elif size_bytes >= 1024 ** 2:
                    cache_size_str = f"{size_bytes / 1024**2:.2f} MB"
                else:
                    cache_size_str = f"{size_bytes / 1024:.1f} KB"
                mtime = os.path.getmtime(self.cache_file)
                cache_mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass

        video_cache_str = self.video_cache_file if self.video_cache_file else "No video cache"
        video_cache_size_str = "N/A"
        if self.video_cache_file and os.path.exists(self.video_cache_file):
            try:
                size_bytes = os.path.getsize(self.video_cache_file)
                if size_bytes >= 1024 ** 3:
                    video_cache_size_str = f"{size_bytes / 1024**3:.2f} GB"
                elif size_bytes >= 1024 ** 2:
                    video_cache_size_str = f"{size_bytes / 1024**2:.2f} MB"
                else:
                    video_cache_size_str = f"{size_bytes / 1024:.1f} KB"
            except Exception:
                pass

        exclusions_str = ", ".join(sorted(self.excluded_folders)) if self.excluded_folders else "None"
        total_images = len(self.image_paths) if self.image_paths else 0
        total_frames = len(self.video_paths) if self.video_paths else 0
        total_videos = len(set(vp for vp, _ in self.video_paths)) if self.video_paths else 0

        info = (
            f"Folder:\n  {folder_str}\n\n"
            f"Image Cache:\n  {cache_str}\n"
            f"Cache Size:  {cache_size_str}\n"
            f"Cache Modified:  {cache_mtime_str}\n\n"
            f"Images Indexed:  {total_images:,}\n\n"
            f"Video Cache:\n  {video_cache_str}\n"
            f"Video Cache Size:  {video_cache_size_str}\n\n"
            f"Videos Indexed:  {total_videos:,} ({total_frames:,} frames)\n\n"
            f"Model:  {MODEL_NAME}\n"
            f"Pretrained:  {MODEL_PRETRAINED}\n\n"
            f"Exclusion Patterns:  {exclusions_str}"
        )
        messagebox.showinfo("Index Info", info)

if __name__ == "__main__":
    print("=" * 60)
    print("Makimus - AI Media Search (Cross-Platform GPU Accelerated)")
    print("With Relative Path Support for Portability")
    print("=" * 60)
    root = tk.Tk()
    app = ImageSearchApp(root)
    root.mainloop()