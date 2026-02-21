# HealthMate

**Fine-tuned TinyLlama-1.1B using LoRA for medical domain Q&A**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Niyobelyse/HealthMate/blob/main/notebooks/medical_chatbot.ipynb)

## Overview & Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Setup Guide](#setup-guide)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
  - [Full Stack Running](#full-stack-running)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Model & Fine-tuning](#model--fine-tuning)
- [Evaluation Results](#evaluation-results)
- [Web Interface](#web-interface)
- [FastAPI Backend](#fastapi-backend)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)


---

## Overview

HealthMate is a **domain-specific medical chatbot** built by fine-tuning TinyLlama-1.1B using Low-Rank Adaptation (LoRA). It features:

-  **Fine-tuned Model**: LoRA-adapted TinyLlama for medical Q&A (0.44% trainable params)
-  **React + Tailwind UI**: Modern, responsive web interface
-  **FastAPI Backend**: RESTful API for model inference
-  **Model Comparison**: Compare base vs fine-tuned responses side-by-side
-  **Evaluation**: BLEU, ROUGE, Perplexity metrics
-  **Visualizations**: Dataset analysis, training results, metrics
-  **Production Ready**: Optimized for CPU inference (~5-10 sec per response)

## Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/Niyobelyse/HealthMate.git
cd HealthMate

# Install backend dependencies
pip install -r requirements.txt

# Install React dependencies
cd chatbot-ui
npm install
cd ..
```

### 2. Run Backend (Terminal 1)
```bash
cd backend
python fastapi_backend.py
# API will start at http://localhost:8001
```

### 3. Run Frontend (Terminal 2)
```bash
cd chatbot-ui
npm run dev
# UI will open at http://localhost:3000
```

### 4. Visit http://localhost:3000 and start chatting! 

---

## Prerequisites

- **Python 3.8+** (for backend)
- **Node.js 16+** (for React frontend)
- **Git** (for cloning repo)
- **4GB+ RAM** (minimum for models)
- **GPU recommended** (for faster training/inference, but CPU works)

**Check versions:**
```bash
python --version
node --version
npm --version
git --version
```

---

## Setup Guide

### Clone Repository

```bash
git clone https://github.com/Niyobelyse/HealthMate.git
cd HealthMate
```

---

### Backend Setup

#### Step 1: Create Virtual Environment

```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**Verify activation:**
```bash
which python  # Linux/Mac
where python  # Windows
```

#### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**This installs:**
- PyTorch + Transformers (model inference)
- FastAPI + Uvicorn (REST API)
- PEFT + BitsAndBytes (fine-tuning)
- NLTK + ROUGE (evaluation metrics)
- Matplotlib + Seaborn (visualizations)

#### Step 3: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import fastapi; print('FastAPI: OK')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

#### Step 4: Start FastAPI Backend

cd backend
```bash
python fastapi_backend.py
```

**Expected output:**
```
Using device: cpu
Loading base model...
Loading fine-tuned model...
✓ Fine-tuned model loaded successfully
✓ API ready!
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
```

**API is now live at: http://localhost:8001**

Test it:
```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"message": "What is diabetes?", "model_type": "fine-tuned"}'
```

---

### Frontend Setup

#### Step 1: Navigate to React Directory

```bash
cd chatbot-ui
```

#### Step 2: Install Node Dependencies

```bash
npm install
```

**This installs:**
- React 18
- Tailwind CSS
- Vite (build tool)
- Axios (HTTP client)

#### Step 3: Start Development Server

```bash
npm run dev
```

**Expected output:**
```
  VITE v5.4.21  ready in 390 ms
  ➜  Local:   http://localhost:3000/
  ➜  press h + enter to show help
```

**UI is now live at: http://localhost:3000**

---

### Full Stack Running

#### Terminal 1: Backend
```bash
cd backend
python fastapi_backend.py
```

#### Terminal 2: Frontend
```bash
cd chatbot-ui
npm run dev
```

#### Terminal 3: Jupyter (Optional)
```bash
cd notebooks
jupyter notebook medical_chatbot.ipynb
```

#### Browser
Open **http://localhost:3000** → Start chatting! 

---

## Dataset & Preprocessing

| Metric | Value |
|--------|-------|
| Dataset | Malikeh1375/medical-question-answering-datasets |
| Raw Examples | 5,000 |
| After Filtering | 4,862 (97.2% valid) |
| Train/Test Split | 90% / 10% |
| Topics | Diabetes, Hypertension, Treatments, Medications, etc. |

**Preprocessing Pipeline:**
1. Text cleaning (lowercase, remove special chars, normalize spaces)
2. Tokenization with AutoTokenizer
3. Chat template formatting (system → user → assistant)
4. Quality filtering (minimum 20 char answers, 10 char questions)

---

## Model & Fine-tuning

### Architecture
- **Base Model**: TinyLlama-1.1B-Chat-v1.0
- **Method**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit NF4 (BitsAndBytes)
- **Memory**: 4.4GB → 1.1GB (75% reduction)
- **Trainable Params**: 4.9M / 1.1B (0.44%)

### Hyperparameter Experiments

| Exp | LoRA r | LR | Batch | Epochs | Loss | GPU | Time |
|-----|--------|-------|-------|--------|------|-----|------|
| 1 | 8 | 2e-4 | 2 | 1 | 1.9826 | 11.84GB | 52min |
| 2 | 16 | 1e-4 | 2 | 2 | 1.7134 | 12.31GB | 110min |
| 3 | 32 | 5e-5 | 4 | 3 | **1.6047** | 13.02GB | 173min |

**Best Config**: Experiment 3 achieved lowest training loss with higher rank and more epochs.

---

## Evaluation Results

### Quantitative Metrics (30 test examples)

| Metric | Base Model | Fine-Tuned | Improvement |
|--------|-----------|-----------|-------------|
| BLEU | 0.1248 | 0.1542 | **+23.6%**  |
| ROUGE-1 | 0.3152 | 0.3847 | **+21.9%**  |
| ROUGE-L | 0.2894 | 0.3604 | **+24.6%**  |
| Perplexity | 47.31 | 31.84 | **-32.7%**  |

### Qualitative Analysis
The fine-tuned model:
- Uses domain-specific medical terminology
- Provides more accurate treatment information
- Handles out-of-domain queries appropriately
- Generates longer, more detailed responses

---

## Web Interface (React + Tailwind)

### Features
- **Chat History**: Full conversation tracking
- **Model Selector**: Compare Base vs Fine-tuned models
- **Example Questions**: Quick-start templates
- **Typing Animation**: Real-time feedback
- **Responsive Design**: Mobile-friendly layout
- **Error Handling**: Clear error messages

### Architecture
```
chatbot-ui/
├── src/
│   ├── App.jsx              (main state & API logic)
│   ├── components/
│   │   ├── Header.jsx       (model selector)
│   │   ├── ChatWindow.jsx   (message display)
│   │   ├── Message.jsx      (individual bubbles)
│   │   ├── InputForm.jsx    (text input)
│   │   └── TypingIndicator.jsx (loading animation)
│   └── index.css            (Tailwind styles)
```

---

## FastAPI Backend

### Endpoints

**POST /query**
```json
{
  "message": "What are symptoms of diabetes?",
  "model_type": "fine-tuned"  // or "base"
}
```

**Response:**
```json
{
  "response": "Symptoms of diabetes include...",
  "model_used": "fine-tuned"
}
```

**GET /health**
Returns API health status and loaded models.

**GET /models**
Lists available models.

### Performance
- **Response Time**: 5-10 seconds (CPU)
- **Max Token Length**: 128 (optimized for speed)
- **Temperature**: 0.5 (focused responses)
- **Sampling**: Greedy decoding (fastest)

---

## Project Structure

```
HealthMate/
├── README.md                          (this file)
├── requirements.txt                   (Python dependencies)
│
├── backend/                           (FastAPI server)
│   └── fastapi_backend.py
│
├── models/                            (Fine-tuned model weights)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── chat_template.jinja
│
├── notebooks/                         (Jupyter notebook)
│   └── medical_chatbot.ipynb
│
└── chatbot-ui/                        (React UI)
    ├── src/
    │   ├── App.jsx                   (main React component)
    │   ├── components/
    │   │   ├── Header.jsx            (model selector)
    │   │   ├── ChatWindow.jsx        (message display)
    │   │   ├── InputForm.jsx         (user input)
    │   │   ├── Message.jsx           (message bubbles)
    │   │   └── TypingIndicator.jsx   (loading animation)
    │   └── index.css                 (styles)
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    └── index.html
```

---

## Jupyter Notebook Cells

| Cell | Section | Purpose |
|------|---------|---------|
| 1-5 | Setup | Imports, device check, Colab compatibility |
| 6-8 | Data | Load, preprocess, filter dataset |
| 9-15 | Exploration | 7 visualizations (tokens, topics, etc.) |
| 16 | Model | Load TinyLlama with 4-bit quantization |
| 17-22 | Training | 3 hyperparameter experiments + best model |
| 23-25 | Evaluation | BLEU, ROUGE, Perplexity metrics |
| 26-32 | Visualizations | Training results, metrics comparison |
| 33-35 | Examples | Qualitative base vs fine-tuned |
| 36+ | UI | Gradio chatbot interface |

---

## Configuration

### Modify Model Response
Edit `backend/fastapi_backend.py`:
```pytbackend/hon
max_new_tokens=50,      # Change response length
temperature=1.0,        # Change randomness (0=deterministic, 1=random)
do_sample=False,        # False=greedy (faster), True=sampling
```

### Change Model
Edit `backend/fastapi_backend.py` line ~47:
```python
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Change this
```

### Customize UI Colors
Edit `chatbot-ui/tailwind.config.js`:
```javascript
colors: {
  medical: {
    600: '#YOUR_COLOR',
    700: '#YOUR_COLOR',
  }
}
```

---

## Technology Stack

**Backend:**
- PyTorch + Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- BitsAndBytes (4-bit quantization)
- FastAPI + Uvicorn
- TRL (Trainer)

**Frontend:**
- React 18
- Tailwind CSS
- Vite (build tool)
- Axios (HTTP requests)

**ML Libraries:**
- NLTK (BLEU scoring)
- ROUGE-Score (metric computation)
- Matplotlib + Seaborn (visualizations)

---

## License

This project is open-source and available under the MIT License.

---

## Author

Built as a fine-tuning project for domain-specific LLMs.

---

## FAQ

**Q: Can I use this for real medical advice?**
A: No. This is educational only. Always consult qualified healthcare professionals.

**Q: How do I improve accuracy?**
A: Add more domain-specific training data or fine-tune with domain experts' Q&A pairs.

**Q: Can this run on my laptop?**
A: Yes, on CPU (~5-10 sec/response). GPU is much faster.

**Q: How do I add more medical domains?**
A: Fine-tune with additional datasets from HuggingFace Hub.

---

## Troubleshooting

### Issue: Port Already in Use

**Error:** `Address already in use`

**Solution:**
```bash
# Find and kill process on port 8001
lsof -i :8001
kill -9 <PID>

# Or use different port - edit backend/fastapi_backend.py:
# Change: uvicorn.run(app, host="0.0.0.0", port=8001)
# To:     uvicorn.run(app, host="0.0.0.0", port=8002)
```

### Issue: Module Not Found

**Error:** `ModuleNotFoundError: No module named 'torch'`

**Solution:**
```bash
# Verify virtual environment is activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate      # Windows

# Reinstall requirements
pip install -r requirements.txt
```

### Issue: Node Modules Missing

**Error:** `Cannot find module 'react'`

**Solution:**
```bash
cd frontend/chatbot-ui
rm -rf node_modules package-lock.json
npm install
```

### Issue: API Not Responding

**Error:** "Failed to fetch from localhost:8001"

**Solution:**
1. Check backend is running: `http://localhost:8001/health`
2. Verify CORS is enabled in `fastapi_backend.py`
3. Check React is running on port 3000
4. Clear browser cache: Ctrl+Shift+R

### Issue: Slow Responses

**Performance Tips:**
- Run on GPU (10x faster)
- Reduce `max_new_tokens` in `backend/fastapi_backend.py`
- Use `do_sample=False` for faster greedy decoding
- Reduce batch size

---

## Environment Variables (Optional)

Create `.env`:
```
BACKEND_URL=http://localhost:8001
MODEL_PATH=/home/belysetag/Desktop/chatbot
LOG_LEVEL=INFO
```

Load in Python:
```python
from dotenv import load_dotenv
import os

load_dotenv()
backend_url = os.getenv("BACKEND_URL", "http://localhost:8001")
```

---

## Model Download

Models are auto-downloaded on first run:

- **Base Model**: TinyLlama-1.1B (~2.2GB) — downloads to `~/.cache/huggingface/`
- **Tokenizer**: ~2MB
- **LoRA Adapter**: Already in repo (`adapter_*.safetensors`)

**Disable download caching:**
```bash
export HF_HUB_OFFLINE=1  # Use cached models only
export HF_HUB_CACHE=/custom/path  # Change cache location
```

---

## Verification Checklist

After setup, verify everything works:

- [ ] Backend running on port 8001
- [ ] Frontend running on port 3000
- [ ] Can open http://localhost:3000 in browser
- [ ] Model selector works (Base vs Fine-tuned)
- [ ] Can send a message and get response
- [ ] Response appears within 10 seconds
- [ ] No errors in browser console (F12)
- [ ] No errors in terminal

---

## Using the Jupyter Notebook

### Option 1: Local Jupyter
```bash
jupyter notebook "medical_chatbot_final_(2).ipynb"
cd notebooks
jupyter notebook medical_chatbot.ipynb

### Option 2: Google Colab (Recommended)
Click the "Open in Colab" badge in README.md

**In Colab:**
1. Run cells 1-5 (setup)
2. Run cells 6-8 (data loading)
3. Run cell 9 (model loading)
4. Skip training cells (use pre-recorded results)
5. Run evaluation & visualization cells

---

## Next Steps

1. **Customize**: Edit colors in `chatbot-ui/tailwind.config.js`
2. **Fine-tune**: Modify hyperparameters in `fastapi_backend.py`
3. **Improve**: Add more training data for better accuracy

---

## Need Help?

1. Check logs: `npm run dev` shows React errors
2. Check API: Visit `http://localhost:8001/docs` (FastAPI Swagger UI)
3. Test endpoint: Use curl or Postman to test API
4. Read errors carefully — they usually indicate the issue

---

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Transformers Library](https://huggingface.co/docs/transformers)
- [PEFT LoRA Guide](https://huggingface.co/docs/peft)
- [Tailwind CSS](https://tailwindcss.com/)

