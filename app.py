import os, pickle, faiss, numpy as np, fitz, time
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq
groq_api_key = os.getenv("GROQ_API_KEY")
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"
embedder = SentenceTransformer(EMBED_MODEL)
client = Groq(api_key=groq_api_key)

# ===== CORE FUNCTIONS =====
def extract_pages_fast(pdf_path):
    doc = fitz.open(pdf_path)
    return [page.get_text("text") for page in doc]

def build_and_save_index(pdf_path, prefix):
    pages = extract_pages_fast(pdf_path)
    embeds = embedder.encode(pages, convert_to_numpy=True, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeds.shape[1])
    index.add(embeds)
    faiss.write_index(index, f"{prefix}.faiss")
    np.save(f"{prefix}_embeds.npy", embeds)
    with open(f"{prefix}_pages.pkl", "wb") as f:
        pickle.dump(pages, f)
    return len(pages)

def load_index(prefix):
    index = faiss.read_index(f"{prefix}.faiss")
    embeds = np.load(f"{prefix}_embeds.npy")
    with open(f"{prefix}_pages.pkl", "rb") as f:
        pages = pickle.load(f)
    return pages, embeds, index

def get_context(page_num, question, pages, embeds, index, k=5):
    start, end = max(0, page_num-10), min(len(pages), page_num+11)
    q_embed = embedder.encode([question], convert_to_numpy=True)
    D, I = index.search(q_embed, k)
    retrieved = [pages[i] for i in I[0] if start <= i < end]
    context = f"=== Focus Page {page_num} ===\n{pages[page_num]}\n\n=== Related Context ===\n" + "\n\n".join(retrieved)
    return context

def ask_llm(context, question):
    prompt = f"""You are a tutor. Use ONLY the text below to answer clearly.

{context}

Question: {question}"""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role":"user","content":prompt}]
    )
    return resp.choices[0].message.content

# ===== STREAMLIT UI =====
st.title("📘 Ask Your PDF (AI Tutor)")
st.caption("Upload once, then ask unlimited questions page-wise.")

pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
prefix = st.text_input("Index name (for saving/loading):", "textbook")

if pdf_file and st.button("Name PDF and ready AI Tutor"):
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.read())
    with st.spinner("Building index... (one-time process)"):
        n_pages = build_and_save_index("temp.pdf", prefix)
    st.success(f"Indexed {n_pages} pages successfully.")

if os.path.exists(f"{prefix}.faiss"):
    st.success("Index found and loaded.")
    pages, embeds, index = load_index(prefix)

    # Ask for the original PDF path (we'll use it to show pages)
    pdf_path =st.text_input("Path to original PDF file (used for display):", "temp.pdf")

    if os.path.exists(pdf_path):
        doc = fitz.open(pdf_path)
        page_num = st.number_input("Page number:", 0, len(pages)-1, 0)

        # --- Display selected PDF page ---
        zoom = 2  # increase for higher resolution
        mat = fitz.Matrix(zoom, zoom)
        pix = doc.load_page(page_num).get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        st.image(img_bytes, caption=f"Page {page_num}", use_container_width=True)

        question = st.text_input("Ask a question about this page:")
        if st.button("Ask"):
            with st.spinner("Generating answer..."):
                ctx = get_context(page_num, question, pages, embeds, index)
                ans = ask_llm(ctx, question)
            st.markdown("### 🧩 Answer")
            st.write(ans)
    else:
        st.warning("Please provide the path to the original PDF file so the page can be displayed.")
