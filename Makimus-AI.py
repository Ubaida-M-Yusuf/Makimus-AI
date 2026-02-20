import os
import sys
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from threading import Thread
import hashlib
from pathlib import Path
import time
import queue
import ctypes
import subprocess
import re
import gc
from PIL import Image, ImageTk
import numpy as np

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

def determine_batch_size():
    vram_bytes = get_system_vram()
    if vram_bytes is None:
        print("[CONFIG] Could not detect VRAM. Defaulting to Batch Size: 16")
        return 16
    
    vram_gb = vram_bytes / (1024**3)
    print(f"[CONFIG] Detected VRAM: {vram_gb:.2f} GB")
    
    if vram_gb >= 15:
        return 256
    elif vram_gb >= 11:
        return 128
    elif vram_gb >= 7:
        return 64
    elif vram_gb >= 5:
        return 32
    else:
        return 16

BATCH_SIZE = determine_batch_size()
print(f"[CONFIG] Selected Batch Size: {BATCH_SIZE}")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")
TOP_RESULTS = 30
MIN_SCORE_THRESHOLD = 0.20
THUMBNAIL_SIZE = (180, 180)
MAX_DISPLAY_RESULTS = 1000
MAX_THUMBNAIL_CACHE = 2000  # Limit RAM usage - clear old thumbnails after this
CELL_WIDTH = 220
CELL_HEIGHT = 260
CACHE_PREFIX = ".clip_cache_"
CACHE_SUFFIX = ".pkl"

MODEL_CONFIGS = {
    "ViT-L-14 (LAION2B)": {
        "name": "ViT-L-14",
        "pretrained": "laion2b_s32b_b82k",
        "image_size": 224,
    },
    "PE-Core-L14-336": {
        "name": "hf-hub:timm/PE-Core-L14-336",
        "pretrained": None,
        "image_size": 336,
    },
    "PE-Core-G14-448": {
        "name": "hf-hub:timm/PE-Core-bigG-14-448",
        "pretrained": None,
        "image_size": 448,
    },
}
DEFAULT_MODEL_KEY = "ViT-L-14 (LAION2B)"
# Legacy references kept for any code that reads them directly
MODEL_NAME = MODEL_CONFIGS[DEFAULT_MODEL_KEY]["name"]
MODEL_PRETRAINED = MODEL_CONFIGS[DEFAULT_MODEL_KEY]["pretrained"]

BG = "#1e1e1e"
PANEL_BG = "#252526"
CARD_BG = "#2d2d30"
FG = "#e0e0e0"
ACCENT = "#4CAF50"
ACCENT_SECONDARY = "#3fa9f5"
DANGER = "#f44336"
ORANGE = "#ff9800"
BORDER = "#3c3c3c"

def safe_print(text, end='\n'):
    try:
        print(text, end=end)
    except:
        pass

class HybridCLIPModel:
    """
    Cross-Platform Hybrid Model Wrapper
    """
    def __init__(self, model_key=None):
        import torch
        import open_clip
        import onnxruntime as ort

        # Resolve model config
        if model_key is None:
            model_key = DEFAULT_MODEL_KEY
        self.model_key = model_key
        config = MODEL_CONFIGS.get(model_key, MODEL_CONFIGS[DEFAULT_MODEL_KEY])
        self.image_size = config["image_size"]
        _model_name = config["name"]
        _model_pretrained = config["pretrained"]

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
        safe_print(f"[MODEL] Loading: {_model_name}")

        # 2. Load PyTorch Model
        model_loaded = False

        try:
            import huggingface_hub
            huggingface_hub.constants.HF_HUB_OFFLINE = True
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

            if _model_pretrained is not None:
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    _model_name, pretrained=_model_pretrained
                )
            else:
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(_model_name)
            self.tokenizer = open_clip.get_tokenizer(_model_name)
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
                safe_print(f"[MODEL] Downloading {_model_name} (this may take a while)...")
                if _model_pretrained is not None:
                    self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                        _model_name, pretrained=_model_pretrained
                    )
                else:
                    self.model, _, self.preprocess = open_clip.create_model_and_transforms(_model_name)
                self.tokenizer = open_clip.get_tokenizer(_model_name)
                safe_print(f"[MODEL] Download complete!")
            except Exception as e:
                safe_print(f"[MODEL] Download failed: {e}")
                raise

        self.model = self.model.to(self.device).eval()

        # 3. Setup ONNX Visual Encoder
        self.setup_onnx_encoder()

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
        _safe_key = self.model_key.replace('hf-hub:', '').replace('timm/', '').replace('/', '_').replace('-', '_').replace(':', '_').replace(' ', '_').replace('(', '').replace(')', '')
        self.onnx_visual_path = cache_dir / f"{_safe_key}_visual.onnx"

        if not self.onnx_visual_path.exists():
            safe_print(f"[ONNX] Attempting visual encoder export...")

            try:
                dummy_image = torch.randn(1, 3, self.image_size, self.image_size).to(self.device)
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
        
        # Skip ONNX on known problematic configurations
        if sys.platform.startswith('linux'):
            # Some Linux CUDA configurations have issues with ONNX export
            try:
                # Quick compatibility test with a tiny model
                test_model = torch.nn.Linear(2, 2).to(self.device)
                test_input = torch.randn(1, 2).to(self.device)
                test_path = Path("/tmp/onnx_test.onnx")
                
                with torch.no_grad():
                    torch.onnx.export(
                        test_model,
                        test_input,
                        test_path,
                        opset_version=14,
                        do_constant_folding=True,
                        verbose=False
                    )
                
                # Clean up test file
                if test_path.exists():
                    test_path.unlink()
                
                return True
                
            except:
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
        target_size = self.image_size
        w, h = image.size
        scale = target_size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
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
        
        # PyTorch fallback (always works)
        try:
            batch_tensors = [self.preprocess_image_pytorch(img) for img in images]
            input_tensor = torch.cat(batch_tensors).to(self.device)
            with torch.no_grad():
                features = self.model.encode_image(input_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
            
            # Move to CPU and explicitly delete GPU tensors
            result = features.cpu().numpy()
            del features, input_tensor, batch_tensors
            
            return result
        except Exception as e:
            safe_print(f"[ERROR] Image encoding failed: {e}")
            import traceback
            traceback.print_exc()
            raise  # Re-raise so caller knows encoding failed
    
    def encode_text(self, texts):
        import torch
        
        try:
            tokens = self.tokenizer(texts).to(self.device)
            with torch.no_grad():
                features = self.model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
            
            # Move to CPU and explicitly delete GPU tensors
            result = features.cpu().numpy()
            del features, tokens
            
            safe_print(f"[ENCODE] Text encoded successfully, shape: {result.shape}")
            return result
        except Exception as e:
            safe_print(f"[ERROR] Text encoding failed: {e}")
            import traceback
            traceback.print_exc()
            raise  # Re-raise so caller knows encoding failed

class ImageSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Makimus - AI Image Search")
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

        self.current_model_key = DEFAULT_MODEL_KEY
        self.clip_model = None
        self.model_loading = False
        
        self.is_indexing = False
        self.stop_indexing = False
        self.is_stopping = False
        
        self.is_searching = False
        self.stop_search = False
        self.search_thread = None
        self.index_thread = None
        self.render_after_id = None
        self.click_timer = None
        
        self.total_found = 0
        self.search_generation = 0
        self.render_cols = 1
        self.thumbnail_queue = queue.Queue()
        
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
        config = MODEL_CONFIGS.get(self.current_model_key, MODEL_CONFIGS[DEFAULT_MODEL_KEY])
        _name = config["name"]
        _pretrained = config.get("pretrained")
        if _pretrained:
            pretrained_simple = "LAION2B" if "laion2b" in _pretrained.lower() else _pretrained.upper()
            cache_name = f".clip_cache_{_name}_{pretrained_simple}.pkl"
        else:
            # Hub models: use a safe filename derived from the hub path
            safe_name = _name.replace("hf-hub:", "").replace("timm/", "").replace("/", "_")
            cache_name = f".clip_cache_{safe_name}.pkl"
        return [cache_name]

    def load_model(self, model_key=None):
        if model_key is None:
            model_key = self.current_model_key
        self.model_loading = True
        self.update_status("Loading model...", "orange")
        try:
            self.clip_model = HybridCLIPModel(model_key=model_key)
            self.root.after(0, lambda: self.update_status("Ready", "green"))
            safe_print(f"[LOAD] Success!\n")
        except Exception as e:
            safe_print(f"[ERROR] {e}")
            self.root.after(0, lambda: self.update_status("Load Failed", "red"))
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load model\n{e}"))
        self.model_loading = False

    def on_reload_model(self):
        if self.model_loading:
            messagebox.showwarning("Wait", "Model is already loading...")
            return
        if not self.is_safe_to_act(action_callback=self._do_reload_model, action_name="reload model"):
            return
        self._do_reload_model()

    def _do_reload_model(self):
        new_key = self.model_var.get()
        if new_key == self.current_model_key and self.clip_model is not None:
            messagebox.showinfo("Info", f"Model '{new_key}' is already loaded.")
            return
        self.current_model_key = new_key
        config = MODEL_CONFIGS[new_key]
        self.model_info_label.config(text=f"img {config['image_size']}px")
        # Clear any cached embeddings — different models are incompatible
        self.image_paths = []
        self.image_embeddings = None
        self.cache_file = None
        self.clear_results()
        self.update_stats()
        # Unload old model to free VRAM before loading new one
        if self.clip_model is not None:
            try:
                import torch, gc
                self.clip_model.model.cpu()
                del self.clip_model
                self.clip_model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            except Exception:
                pass
        Thread(target=lambda: self.load_model(model_key=new_key), daemon=True).start()

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
        
        self.btn_clear = ttk.Button(top, text="Clear Cache", command=self.on_delete_cache, width=12, style="Danger.TButton")
        self.btn_clear.pack(side="left", padx=4)
        
        ttk.Label(top, text=f"Using: {MODEL_NAME}", foreground=ACCENT_SECONDARY).pack(side="left", padx=(16, 4))
        
        self.btn_stop = ttk.Button(top, text="STOP INDEX", command=self.stop_indexing_process, width=12, style="Danger.TButton")
        self.btn_stop.pack(side="left", padx=(20, 4))

        ttk.Button(top, text="EXIT", command=self.force_quit, width=12, style="Danger.TButton").pack(side="left", padx=6)

        self.status_label = ttk.Label(top, text="Starting...", width=35, anchor="w")
        self.status_label.pack(side="left", padx=10)
        self.stats_label = ttk.Label(top, text="")
        self.stats_label.pack(side="left")

        # Model selector row
        model_row = ttk.Frame(self.root, padding=(8, 0, 8, 4))
        model_row.pack(fill="x", padx=8)
        ttk.Label(model_row, text="Model:").pack(side="left", padx=(0, 4))
        self.model_var = tk.StringVar(value=self.current_model_key)
        self.model_combo = ttk.Combobox(
            model_row, textvariable=self.model_var,
            values=list(MODEL_CONFIGS.keys()), state="readonly", width=28
        )
        self.model_combo.pack(side="left", padx=4)
        ttk.Button(model_row, text="Load Model", command=self.on_reload_model, width=12, style="Accent.TButton").pack(side="left", padx=4)
        self.model_info_label = ttk.Label(model_row, text=f"img {MODEL_CONFIGS[DEFAULT_MODEL_KEY]['image_size']}px", foreground=ACCENT_SECONDARY)
        self.model_info_label.pack(side="left", padx=6)

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
        
        ttk.Label(ctrl_frame, text="Min Score: ").pack(side="left", padx=(0, 4))
        self.score_var = tk.DoubleVar(value=MIN_SCORE_THRESHOLD)
        tk.Scale(ctrl_frame, from_=0.0, to=0.5, resolution=0.05, orient="horizontal", 
                 variable=self.score_var, length=140, bg=PANEL_BG, fg=FG, troughcolor=BORDER, highlightthickness=0).pack(side="left")
        
        ttk.Label(ctrl_frame, text="Max Results:").pack(side="left", padx=(16, 4))
        self.top_n_var = tk.IntVar(value=TOP_RESULTS)
        tk.Scale(ctrl_frame, from_=10, to=MAX_DISPLAY_RESULTS, resolution=10, orient="horizontal", 
                 variable=self.top_n_var, length=170, bg=PANEL_BG, fg=FG, troughcolor=BORDER, highlightthickness=0).pack(side="left")
        
        ttk.Button(ctrl_frame, text="Clear Results", command=self.on_clear_click, width=12).pack(side="left", padx=14)
        ttk.Button(ctrl_frame, text="Export Selected", command=self.on_export_click, width=14).pack(side="left", padx=4)

        self.progress = ttk.Progressbar(self.root, mode='determinate')
        self.progress.pack(fill="x", padx=10, pady=6)
        self.progress_label = ttk.Label(self.root, text="", anchor="center")
        self.progress_label.pack()

        results_container = ttk.Frame(self.root, padding=6)
        results_container.pack(fill="both", expand=True, padx=8, pady=6)
        
        self.canvas = tk.Canvas(results_container, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(results_container, orient="vertical", command=self.canvas.yview)
        
        self.results_frame = tk.Frame(self.canvas, bg=BG, highlightthickness=0)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.results_frame, anchor="nw")
        
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.bind('<Configure>', self.on_canvas_configure)
        self.results_frame.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        if sys.platform == 'darwin':
            self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-1 * e.delta), "units"))
        else:
            self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

    def is_safe_to_act(self, action_callback=None, action_name="action"):
        """
        Returns True if no indexing is happening.
        If indexing is running, automatically stops it and queues the action.
        """
        if self.is_indexing or self.is_stopping:
            if not self.is_stopping:
                self.pending_action = action_callback
                self.stop_indexing_process()
                self.update_status(f"Stopping to {action_name}... Please wait", "orange")
                self.btn_stop.config(text="STOP (Cancel Pending)")
            else:
                self.pending_action = action_callback
                self.update_status(f"Stopping in progress... Will {action_name} when done", "orange")
                self.btn_stop.config(text="STOP (Cancel Pending)")
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

    def on_export_click(self):
        self.export_selected()

    def on_search_click(self):
        self.cancel_search(clear_ui=True)
        self.do_search()

    def on_image_click(self):
        self.cancel_search(clear_ui=True)
        self.image_search()

    def on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def force_quit(self):
        if messagebox.askyesno("Force Quit", "Force quit application?"):
            os._exit(0)

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
        if self.render_after_id:
            self.root.after_cancel(self.render_after_id)
            self.render_after_id = None
        
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
        
        self.image_paths = []
        self.image_embeddings = None
        self.folder = None
        self.cache_file = None
        self.clear_results()
        self.update_stats()
        
        folder = filedialog.askdirectory()
        if not folder:
            self.update_status("No folder selected", "orange")
            return
        
        self.folder = folder
        safe_print(f"\n{'='*60}\n[FOLDER] {folder}")
        
        cache_files = self.get_cache_filename()
        found_cache = None
        for cache_name in cache_files:
            cache_path = os.path.join(folder, cache_name)
            if os.path.exists(cache_path):
                found_cache = cache_path
                safe_print(f"[CACHE] Found existing: {cache_name}")
                break
        
        if found_cache:
            self.cache_file = found_cache
            self.load_cache_data(found_cache)
            query = self.query_entry.get().strip()
            if query:
                self.root.after(500, self.do_search)
        else:
            safe_print("[CACHE] Not found")
            if messagebox.askyesno("Index Folder?", f"No cache found for this folder.\n\nIndex images now?"):
                self.cache_file = os.path.join(folder, cache_files[0])
                self.start_indexing(mode="full")
            else:
                self.update_status("Folder loaded (Not indexed)", "orange")

    def select_cache(self):
        cache = filedialog.askopenfilename(filetypes=[("Pickle", "*.pkl")])
        if not cache: return
        self.load_cache_data(cache)
        query = self.query_entry.get().strip()
        if query:
            self.root.after(500, self.do_search)

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
            
            self.update_stats()
            self.update_status(f"Loaded {len(self.image_paths):,} images", "green")
            safe_print(f"[CACHE] Success. {len(self.image_paths):,} images (relative paths).")
        except Exception as e:
            messagebox.showerror("Error", f"Load failed: {e}")
            self.update_status("Cache load failed", "red")

    def start_indexing(self, mode="full"):
        self.stop_indexing = False
        self.is_stopping = False
        self.pending_action = None
        self.update_status("Indexing...", "orange")
        self.btn_stop.config(text="STOP INDEX")
        
        if mode == "full":
            self.index_thread = Thread(target=self.index_all_images, daemon=True)
        elif mode == "refresh":
            self.index_thread = Thread(target=self.refresh_index, daemon=True)
        
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
        
        # Build set of current disk files (relative paths)
        for root, _, files in os.walk(self.folder):
            if self.stop_indexing: break
            for f in files:
                if f.lower().endswith(IMAGE_EXTS):
                    abs_path = os.path.join(root, f)
                    rel_path = os.path.relpath(abs_path, self.folder)
                    current_disk_files.add(rel_path)
                    if rel_path not in self.image_paths:
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
                self._save_cache()
            
            self.is_indexing = False
            self.is_stopping = False
            safe_print("[SCAN] Index is up to date.")
            self.root.after(0, lambda: self.update_status("Up to date", "green"))
            self.root.after(0, self.update_stats)

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
        
        self.image_paths = []
        self.image_embeddings = None
        
        all_images = []
        for root, _, files in os.walk(self.folder):
            if self.stop_indexing: break
            for f in files:
                if f.lower().endswith(IMAGE_EXTS):
                    all_images.append(os.path.join(root, f))
        
        if not all_images:
            self.is_indexing = False
            self.root.after(0, lambda: self.update_status("No images found", "orange"))
            return

        safe_print(f"[INDEX] Found {len(all_images)} images.")
        self._process_batch(all_images, is_update=False)

    def _process_batch(self, file_list, is_update=False):
        """Process images and store RELATIVE paths"""
        try:
            total = len(file_list)
            processed = 0
            
            for i in range(0, total, BATCH_SIZE):
                if self.stop_indexing:
                    safe_print("\n[INDEX] Stopping batch loop.")
                    break
                    
                batch_paths = file_list[i:i + BATCH_SIZE]
                pil_images = []
                valid_batch_paths = []
                
                for abs_path in batch_paths:
                    if self.stop_indexing: break
                    try:
                        img = Image.open(abs_path)
                        # Handle palette images with transparency properly
                        if img.mode == 'P' and 'transparency' in img.info:
                            img = img.convert("RGBA").convert("RGB")
                        elif img.mode != 'RGB':
                            img = img.convert("RGB")
                        pil_images.append(img)
                        valid_batch_paths.append(abs_path)
                    except:
                        pass
                
                if self.stop_indexing: break
                
                if pil_images:
                    try:
                        safe_print(f"[INDEX] Encoding batch of {len(pil_images)} images...")
                        features = self.clip_model.encode_image_batch(pil_images)
                        
                        if features is None or features.size == 0:
                            safe_print(f"[INDEX ERROR] Encoding returned empty array, skipping batch")
                            continue
                        
                        # Convert absolute paths to relative before storing
                        existing_paths_set = set(self.image_paths)
                        new_features = []
                        new_paths = []
                        
                        for idx, abs_path in enumerate(valid_batch_paths):
                            rel_path = os.path.relpath(abs_path, self.folder)
                            if rel_path not in existing_paths_set:
                                new_paths.append(rel_path)  # Store RELATIVE path
                                new_features.append(features[idx])
                        
                        # Only add if we have new unique paths
                        if new_paths:
                            self.image_paths.extend(new_paths)
                            new_features_array = np.array(new_features)
                            
                            if self.image_embeddings is None:
                                self.image_embeddings = new_features_array
                            else:
                                self.image_embeddings = np.concatenate([self.image_embeddings, new_features_array])
                            
                            processed += len(new_paths)
                            safe_print(f"[INDEX] Successfully encoded {len(new_paths)} images")
                        
                    except Exception as e:
                        safe_print(f"[ERROR] Batch encoding failed: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Clear PIL images from memory
                del pil_images, valid_batch_paths
                
                # Clear GPU cache EVERY batch to prevent VRAM accumulation
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()

                pct = (processed / total) * 100 if total > 0 else 0
                msg = f"{'Updating' if is_update else 'Indexing'}: {processed:,}/{total:,}"
                self.root.after(0, lambda v=pct, m=msg: self.update_progress(v, m))
                safe_print(f"\r[INDEX] {msg}", end='')

            safe_print("")
            
        finally:
            # Force aggressive VRAM cleanup at end
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            # Force garbage collection to free any temporary memory
            gc.collect()
        
        self._save_cache()
        self._handle_stop()

    def _save_cache(self):
        """Save cache with RELATIVE paths"""
        if self.image_embeddings is not None and len(self.image_paths) > 0:
            try:
                temp_file = self.cache_file + ".tmp"
                with open(temp_file, "wb") as f:
                    # Save only relative paths and embeddings
                    pickle.dump((self.image_paths, self.image_embeddings), f)
                if os.path.exists(self.cache_file):
                    os.remove(self.cache_file)
                os.rename(temp_file, self.cache_file)
                safe_print(f"[CACHE] Saved {len(self.image_paths)} relative paths to {self.cache_file}")
            except Exception as e:
                safe_print(f"[CACHE] Save Error: {e}")

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
        
        self.root.after(0, lambda: self.btn_stop.config(text="STOP INDEX"))
        self.root.after(0, lambda: self.progress.configure(value=0))
        self.root.after(0, lambda: self.progress_label.config(text=""))
        self.root.after(0, self.update_stats)

        if was_stopped:
            msg = f"Stopped. Saved {count:,} images."
            safe_print(f"[INDEX] {msg}")
            self.root.after(0, lambda: self.update_status(msg, DANGER))
            
            if self.pending_action:
                safe_print("[ACTION] Executing pending action...")
                action = self.pending_action
                self.pending_action = None
                self.root.after(100, action)
            elif count > 0:
                query = self.query_entry.get().strip()
                if query:
                    self.root.after(500, self.do_search)
        else:
            self.root.after(0, lambda: self.update_status("Indexing Complete", "green"))
            self.root.after(0, lambda: messagebox.showinfo("Done", f"Index complete.\nTotal images: {count:,}"))
            query = self.query_entry.get().strip()
            if query:
                self.root.after(500, self.do_search)

    def delete_cache(self):
        if not self.folder: return
        if not messagebox.askyesno("Clear?", "Delete cache file and re-index everything?"): return
        
        try:
            if self.cache_file and os.path.exists(self.cache_file):
                os.remove(self.cache_file)
                safe_print("[CACHE] Deleted.")
        except: pass
        
        self.image_paths = []
        self.image_embeddings = None
        self.clear_results()
        self.update_stats()
        self.start_indexing(mode="full")

    def force_reindex(self):
        if not self.folder:
            messagebox.showwarning("Warning", "Select a folder first.")
            return
        self.start_indexing(mode="refresh")

    def do_search(self):
        if self.is_searching or self.clip_model is None: 
            safe_print("[SEARCH] Already searching or model not loaded")
            return
        if self.image_embeddings is None or len(self.image_paths) == 0:
            messagebox.showwarning("No Data", "Index is empty. Please select a folder.")
            return
        
        query = self.query_entry.get().strip()
        if not query: 
            safe_print("[SEARCH] Empty query")
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
        self.stop_search = False
        
        safe_print(f"[SEARCH] Generation: {generation}, Query: '{query}'")
        
        # Clear old results and free RAM before new search
        self.root.after(0, self.clear_results)
        self.total_found = 0
        
        if not self.is_indexing:
            self.root.after(0, lambda: self.update_status("Searching...", "orange"))
            self.root.after(0, lambda: self.progress.config(mode='indeterminate'))
            self.root.after(0, self.progress.start)
            
        try:
            safe_print(f"[SEARCH] Encoding query...")
            text_embed = self.clip_model.encode_text([query])
            
            if text_embed is None or text_embed.size == 0:
                safe_print("[SEARCH ERROR] Text encoding returned empty array")
                self.root.after(0, lambda: self.update_status("Search failed - text encoding error", "red"))
                self.is_searching = False
                return
            
            if self.stop_search or generation != self.search_generation: 
                safe_print("[SEARCH] Cancelled after text encoding")
                return
            
            safe_print(f"[SEARCH] Computing similarities...")
            sims = (self.image_embeddings @ text_embed.T).flatten()
            
            if self.stop_search: 
                safe_print("[SEARCH] Cancelled after similarity computation")
                return
            
            if not self.is_indexing:
                self.root.after(0, self.progress.stop)
                self.root.after(0, lambda: self.progress.config(mode='determinate'))
            
            min_score = self.score_var.get()
            indices = np.where(sims >= min_score)[0]
            
            safe_print(f"[SEARCH] Found {len(indices)} results above threshold {min_score}")
            
            if len(indices) > 0:
                scores = sims[indices]
                sorted_idx = np.argsort(scores)[::-1]
                max_res = min(self.top_n_var.get(), MAX_DISPLAY_RESULTS)
                sorted_idx = sorted_idx[:max_res]
                
                # Convert relative paths to absolute for display
                results = []
                for i in sorted_idx:
                    rel_path = self.image_paths[indices[i]]
                    abs_path = os.path.join(self.folder, rel_path)
                    results.append((scores[i], abs_path))
                
                self.total_found = len(indices)
                
                safe_print(f"[SEARCH] Displaying top {len(results)} results")
                
                cw = max(self.canvas.winfo_width(), CELL_WIDTH)
                self.render_cols = max(1, cw // CELL_WIDTH)
                
                self.start_thumbnail_loader(results, generation)
            else:
                safe_print("[SEARCH] No matches found")
                if not self.is_indexing:
                    self.root.after(0, lambda: self.update_status("No matches found", "green"))
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
        self.stop_search = False
        
        # Clear old results and free RAM before new search
        self.root.after(0, self.clear_results)
        
        cw = max(self.canvas.winfo_width(), CELL_WIDTH)
        self.render_cols = max(1, cw // CELL_WIDTH)
        
        self.root.after(0, lambda: self.update_status("Searching by image...", "orange"))
        
        try:
            img = Image.open(path)
            # Handle palette images with transparency properly
            if img.mode == 'P' and 'transparency' in img.info:
                img = img.convert("RGBA").convert("RGB")
            elif img.mode != 'RGB':
                img = img.convert("RGB")
            features = self.clip_model.encode_image_batch([img])
            emb = features[0]
            
            sims = (self.image_embeddings @ emb).flatten()
            
            min_score = self.score_var.get()
            indices = np.where(sims >= min_score)[0]
            
            if len(indices) > 0:
                scores = sims[indices]
                sorted_idx = np.argsort(scores)[::-1]
                max_res = min(self.top_n_var.get(), MAX_DISPLAY_RESULTS)
                sorted_idx = sorted_idx[:max_res]
                
                # Convert relative paths to absolute for display
                results = []
                for i in sorted_idx:
                    rel_path = self.image_paths[indices[i]]
                    abs_path = os.path.join(self.folder, rel_path)
                    results.append((scores[i], abs_path))
                
                self.total_found = len(indices)
                self.start_thumbnail_loader(results, generation)
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
        Thread(target=self.load_thumbnails_worker, args=(results, generation), daemon=True).start()
        self.root.after(10, lambda: self.check_thumbnail_queue(generation))

    def load_thumbnails_worker(self, results, generation):
        loaded = 0
        failed = 0
        for score, path in results:
            if self.stop_search or generation != self.search_generation: 
                safe_print(f"[THUMBNAILS] Stopped (loaded {loaded}, failed {failed})")
                break
            try:
                img = Image.open(path)
                # Handle palette images with transparency
                if img.mode == 'P' and 'transparency' in img.info:
                    img = img.convert("RGBA")
                img.thumbnail(THUMBNAIL_SIZE)
                img.load()
                self.thumbnail_queue.put((score, path, img))
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
                self.add_result_thumbnail(*item)
                processed_this_cycle += 1
            except queue.Empty: break
            
            if time.time() - start_time > 0.02: break
        
        done = len(self.results_frame.winfo_children())
        target = min(self.total_found, self.top_n_var.get())
        
        if processed_this_cycle > 0:
            safe_print(f"[THUMBNAILS] Displayed {done}/{target} results", end='\r')
        
        if target > 0 and not self.is_indexing:
            self.progress_label.config(text=f"Showing {done} / {self.total_found:,}")
            
        if done >= target or (self.thumbnail_queue.empty() and not self.search_thread.is_alive()):
            if done >= target:
                self.is_searching = False
                if not self.is_indexing:
                    safe_print(f"\n[THUMBNAILS] Display complete: {done} results shown")
                    self.update_status(f"Found {self.total_found:,} results", "green")
                return
        
        self.root.after(10, lambda: self.check_thumbnail_queue(generation))

    def add_result_thumbnail(self, score, path, pil_img):
        if self.stop_search: return
        
        # Limit thumbnail cache size to prevent RAM overflow
        if len(self.thumbnail_images) > MAX_THUMBNAIL_CACHE:
            safe_print(f"[RAM] Thumbnail cache limit reached ({MAX_THUMBNAIL_CACHE}), clearing oldest...")
            # Keep only the most recent thumbnails
            keys_to_remove = list(self.thumbnail_images.keys())[:-MAX_THUMBNAIL_CACHE//2]
            for key in keys_to_remove:
                del self.thumbnail_images[key]
            gc.collect()
        
        cols = max(1, getattr(self, "render_cols", 1))
        idx = len(self.results_frame.winfo_children())
        row, col = divmod(idx, cols)
        
        f = tk.Frame(self.results_frame, bg=CARD_BG, bd=1, relief="solid")
        f.grid(row=row, column=col, padx=6, pady=6)
        f.configure(width=CELL_WIDTH, height=CELL_HEIGHT)
        f.pack_propagate(False)
        
        try:
            photo = ImageTk.PhotoImage(pil_img)
            self.thumbnail_images[path] = photo
            
            lbl = tk.Label(f, image=photo, bg=CARD_BG)
            lbl.pack(pady=4)
            
            lbl.bind("<Button-1>", lambda e: self.handle_single_click(path))
            lbl.bind("<Double-Button-1>", lambda e: self.handle_double_click(path))
            
            var = tk.BooleanVar()
            cb = tk.Checkbutton(f, text="Select", variable=var, bg=CARD_BG, fg=FG, 
                                selectcolor=BG, command=lambda: self.toggle_selection(path, var.get()))
            cb.pack()
            
            name = os.path.basename(path)
            if len(name) > 25: name = name[:22] + "..."
            tk.Label(f, text=f"{score:.3f}\n{name}", bg=CARD_BG, fg=FG, 
                     font=("Segoe UI", 9), wraplength=180, justify="center").pack(pady=2)
        except:
            f.destroy()

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
                subprocess.Popen(f'explorer /select,"{path}"')
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

    def clear_results(self):
        """Clear search results and free RAM from thumbnails"""
        if self.render_after_id:
            self.root.after_cancel(self.render_after_id)
            self.render_after_id = None
        
        # Destroy all thumbnail widgets
        for w in self.results_frame.winfo_children():
            w.destroy()
        
        # Explicitly delete all thumbnail image references
        if self.thumbnail_images:
            for key in list(self.thumbnail_images.keys()):
                del self.thumbnail_images[key]
            self.thumbnail_images.clear()
        
        # Force garbage collection to free RAM immediately
        gc.collect()
        
        self.canvas.yview_moveto(0)
        self.total_found = 0

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
        from shutil import copy2
        count = 0
        for path in self.selected_images:
            try:
                copy2(path, export_dir)
                count += 1
            except: pass
        messagebox.showinfo("Success", f"Exported {count} images")
        self.selected_images.clear()

    def update_status(self, text, color="blue"):
        self.status_label.config(text=text, foreground=color)

    def update_stats(self):
        if self.image_embeddings is not None:
            count = len(self.image_paths)
            self.stats_label.config(text=f"{count:,} images indexed")
        else:
            self.stats_label.config(text="")

    def update_progress(self, value, text):
        self.progress['value'] = value
        self.progress_label.config(text=text)

if __name__ == "__main__":
    print("=" * 60)
    print("AI Image Search (Cross-Platform GPU Accelerated)")
    print("With Relative Path Support for Portability")
    print("=" * 60)
    root = tk.Tk()
    app = ImageSearchApp(root)
    root.mainloop()