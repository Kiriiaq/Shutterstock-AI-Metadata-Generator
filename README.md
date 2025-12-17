# Shutterstock AI Metadata Generator

> **Automate your stock photography workflow with AI-powered image analysis and metadata generation for Shutterstock**

[![Release](https://img.shields.io/badge/release-v1.0.0-blue.svg)](https://github.com/YOUR_USERNAME/shutterstock-ai-metadata-generator/releases)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/AI-Ollama%20Vision-orange.svg)](https://ollama.ai)

---

## The Problem

Stock photographers spend **hours manually writing descriptions, keywords, and categories** for each image before uploading to Shutterstock. This repetitive task is:
- Time-consuming (5-10 minutes per image)
- Error-prone (inconsistent keywords, missing categories)
- Frustrating (takes time away from actual photography)

## The Solution

**Shutterstock AI Metadata Generator** uses local AI vision models (via Ollama) to automatically:
- **Analyze your images** and understand their content
- **Generate SEO-optimized descriptions** (max 200 characters)
- **Create relevant keywords** (7-50 per image)
- **Assign Shutterstock categories** automatically
- **Detect editorial/illustration flags**
- **Export everything to CSV** ready for Shutterstock upload

**No cloud API costs** - Everything runs locally on your computer!

---

## Features

### AI-Powered Analysis
- Uses Ollama vision models (LLaMA 3.2 Vision, LLaVA, Moondream)
- Automatic GPU detection and optimization (NVIDIA CUDA)
- Generates Shutterstock-compliant metadata in seconds

### Smart Image Management
- **Pre-filtering**: Validates images meet Shutterstock requirements (4+ MP, correct format)
- **Batch processing**: Organizes images into folders of 50 (Shutterstock limit)
- **Resume capability**: Continue interrupted processing sessions
- **Duplicate detection**: Avoids reprocessing already analyzed images

### Built-in Ollama Management
- One-click Ollama server start/stop
- **Auto-repair**: Fixes common Ollama issues (zombie processes, port conflicts)
- Model download and loading from the GUI
- Real-time GPU/VRAM status display

### Validation & Upload
- Checklist validator (photos vs metadata matching)
- Metadata completeness verification
- FTPS upload to Shutterstock servers with progress tracking

---

## Quick Start

### Prerequisites
1. **Windows 10/11** (64-bit)
2. **[Ollama](https://ollama.ai/download)** installed
3. **GPU recommended** (NVIDIA with 4GB+ VRAM)

### Installation

#### Option 1: Download Executable (Recommended)
1. Download from [Releases](https://github.com/YOUR_USERNAME/shutterstock-ai-metadata-generator/releases):
   - `ShutterstockAI-MetadataGenerator-v1.0.0.exe` (Release)
   - `ShutterstockAI-MetadataGenerator-v1.0.0_debug.exe` (Debug with console)
2. Double-click to run - no installation needed!

#### Option 2: Run from Source
```bash
git clone https://github.com/YOUR_USERNAME/shutterstock-ai-metadata-generator.git
cd shutterstock-ai-metadata-generator
pip install -r requirements.txt
python shutterstock_analyzer_unified.py
```

### First Run
1. **Start Ollama** - Click "Start" or let auto-repair handle it
2. **Download a model** - Select `llama3.2-vision:11b` and click "Download"
3. **Select your photo folder** - Browse to your images
4. **Click "Start Analysis"** - Watch the AI work!
5. **Find your CSV** in the `Shutterstock/` folder

---

## Workflow

```
1. Photos Folder    →  2. Pre-filter     →  3. AI Analysis
   (your images)        (Valid/Invalid)       (metadata.csv)
                                                   ↓
4. Validation       ←  5. Shutterstock/  ←  Batch folders
   (Checklist tab)       (organized)         (max 50 images)
```

### Folder Structure After Processing
```
Your_Photos/
├── Valid/              # Pre-filtered valid images
├── Invalid/            # Rejected images (too small, wrong format)
├── Shutterstock/       # Batch 1 (up to 50 images + metadata.csv)
├── Shutterstock_2/     # Batch 2 (if needed)
└── Shutterstock_3/     # And so on...
```

---

## Supported Models

| Model | VRAM | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| `moondream:1.8b` | 2GB | Fast | Basic | CPU or low VRAM |
| `llava:7b` | 4GB | Fast | Good | Budget GPUs |
| `llama3.2-vision:11b` | 7GB | Medium | Excellent | **Recommended** |
| `llava:34b` | 20GB | Slow | Best | High-end GPUs |

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 64-bit | Windows 11 |
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| RAM | 8 GB | 16 GB |
| GPU | None (CPU mode) | NVIDIA RTX 3060+ (8GB VRAM) |
| Storage | 500 MB | 20 GB (for AI models) |

---

## Troubleshooting

### Ollama Won't Start
Click the **"Repair"** button - it automatically:
- Kills zombie processes
- Frees port 11434
- Cleans temp files
- Restarts the server

### "No connection" Error
1. Check if Ollama is installed: `ollama --version`
2. Try manual start: `ollama serve`
3. Check Windows Firewall settings

### Slow Performance
- Use a lighter model (`moondream:1.8b`)
- Increase cooldown time between images
- Close other GPU-intensive applications

---

## Known Limitations

- Windows only (macOS/Linux support planned)
- Requires Ollama to be installed separately
- Large images (>100MP) are automatically rejected
- Processing speed depends on GPU/model choice

---

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**Emmanuel Grolleau**

## Acknowledgments

- [Ollama](https://ollama.ai) - Local AI model server
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) - Modern GUI framework
- [Pillow](https://python-pillow.org/) - Image processing

---

## Keywords

`shutterstock metadata generator` `stock photography automation` `AI image analyzer` `photo keywording tool` `Ollama vision` `image description generator` `automatic photo tagging` `Shutterstock CSV generator` `stock photo workflow`

---

**If this tool saves you time, please star the repository!**
