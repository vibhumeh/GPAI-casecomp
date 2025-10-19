# 📘 GPAI Case Competition PDF Q&A App

This app lets you upload a PDF (e.g., textbook or notes) and ask questions about any page.
It uses RAG (Retrieval-Augmented Generation) with Groq’s LLaMA-3 model and FAISS for contextual retrieval.

### ⚙️ 1. Get Your Groq API Key

Go to https://console.groq.com

Sign in → go to the API Keys section.

Create a new key (starts with gsk_...).

Copy it — you’ll need it in your terminal.

Set it as an environment variable

macOS / Linux (zsh, bash):

export GROQ_API_KEY="gsk_your_key_here"


Windows PowerShell:

setx GROQ_API_KEY "gsk_your_key_here"


Tip: To make it permanent, add that line to your ~/.zshrc or ~/.bashrc.
Verify with echo $GROQ_API_KEY.

### 📦 2. Install Dependencies

Clone the repo and install requirements:

git clone https://github.com/vibhumeh/GPAI-casecomp.git
cd GPAI-casecomp
pip install -r requirements.txt


If you’re using a virtual environment (recommended):

python -m venv venv
source venv/bin/activate   # macOS/Linux
### or
venv\Scripts\activate      # Windows

### 🚀 3. Run the App
streamlit run app.py


Then open http://localhost:8501
 in your browser.

🧠 How It Works

Upload a PDF (stored as temp.pdf)

The app parses it once, builds a FAISS index of page embeddings

You can then pick a page, see the real PDF page preview, and ask any question

Groq’s llama-3.3-70b-instantaneous answers using the current page + ±10 context pages, we used this model for the best speed and free API.
paid API and more prompts can improve our proof of concept heavily.


All embeddings are stored locally (.faiss, .npy, .pkl).

If the app can’t find a key, it’ll prompt you to paste it manually.
