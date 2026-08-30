# Plant Disease Multi-Agent AI MVP

GPU-free local Streamlit MVP using NVIDIA-hosted inference.

## Architecture
Image -> NVIDIA Vision Agent -> PlantVillage RAG/FAISS -> Nemotron 3 Super Recommendation Agent -> Streamlit result.

## Setup

### 1. Create environment
Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install
```powershell
python -m pip install -r requirements.txt
```

### 3. Configure NVIDIA
Copy `.env.example` to `.env` and set `NVIDIA_API_KEY`.

### 4. Build PlantVillage index
```powershell
python rag/build_index.py
```

### 5. Run
```powershell
streamlit run app.py
```

Open http://localhost:8501

## Notes
- NVIDIA inference is remote; no local GPU is required.
- PlantVillage labels are used as the initial MVP retrieval corpus.
- For production, add curated agricultural/organic-management documents to the RAG corpus and evaluate disease accuracy on a held-out test set.
