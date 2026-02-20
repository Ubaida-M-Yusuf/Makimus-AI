# 🦉 Makimus-AI - AI Image Search

Search your entire image library using **natural language or an image** — just type what you're looking for or drop an image, and Makimus-AI finds visually similar results instantly using AI.

> Example: type *"Girl wearing a red armor"* or *"dog playing in grass"* — or just **Use an image** to find visually similar ones from your folders.

![Makimus-AI Demo](demo.gif)

---

## ⚙️ Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 12.1 (recommended)
- Windows / Linux / macOS

---

## 🚀 Installation

### 1. Clone the repo
```bash
git clone https://github.com/Ubaida-M-Yusuf/Makimus-AI.git
cd Makimus-AI
```

### 2. Create a virtual environment
python -m venv venv

### 3. Activate the virtual environment

**Windows (Git Bash):**
source venv/Scripts/activate

**Linux / macOS:**
source venv/bin/activate

### 4. Install dependencies
pip install -r requirements.txt

> **Note:** requirements.txt installs PyTorch for CUDA 12.1+. 
> If you have CUDA 11.8 or older, visit 
> [pytorch.org](https://pytorch.org/get-started/locally) 
> to get the correct version for your system.

---

## ▶️ Run

> **Important:** Make sure to activate your virtual environment before running!
> ```bash
> source venv/Scripts/activate  # Windows
> source venv/bin/activate       # Mac/Linux
> ```

python Makimus-AI.py

---

## 📦 Dependencies

| Package | Version |
|---|---|
| Pillow | 12.0.0 |
| numpy | 2.2.6 |
| torch (CUDA 12.1) | 2.5.1+cu121 |
| torchvision (CUDA 12.1) | 0.20.1+cu121 |
| open_clip_torch | 3.2.0 |
| onnxruntime-gpu | 1.23.2 |

---
## 🧠 Model

This app uses **OpenCLIP ViT-L-14** trained on LAION-2B dataset by OpenAI/LAION.

| Property | Value |
|---|---|
| Model | ViT-L-14 |
| Pretrained on | laion2b_s32b_b82k |
| Library | open_clip_torch |

---


## 💡 Features

- 🔍 Natural language image search
- 🖼️ Image-to-image search — find visually similar images
- 🔄 Search while indexing — no need to wait for indexing to finish
- ⚡ GPU accelerated (CUDA, Apple MPS, DirectML)
- 🧠 Auto-detects VRAM and adjusts batch size
- 📦 ONNX optimization for faster inference
- 🖼️ Thumbnail preview with export support
- 💾 Smart caching — index once, search forever

---

## 📝 Notes

- First run will download the AI model (~1GB) automatically
- After indexing your folder, results are cached for instant future searches
- CPU mode works but is significantly slower

---

## 🤝 Contributing

This project is not open for direct contributions.
If you'd like to improve it, feel free to **fork** the repo 
and build your own version!

[![Fork](https://img.shields.io/github/forks/Ubaida-M-Yusuf/Makimus-AI?style=social)](https://github.com/Ubaida-M-Yusuf/Makimus-AI/fork)