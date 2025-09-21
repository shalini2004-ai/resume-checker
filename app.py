# app.py
import streamlit as st
import fitz  # PyMuPDF
import docx2txt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Resume Relevance Checker", layout="wide")
st.title("📄 Automated Resume Relevance Check System")
st.write("Upload a Job Description and resumes to see relevance scores!")

# Load embedding model (cached)
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
model = load_model()

# ------------------------
# File parsing functions
# ------------------------
def parse_pdf(file):
    text = ""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

def parse_docx(file):
    return docx2txt.process(file)

def parse_file(file):
    if file.name.endswith(".pdf"):
        return parse_pdf(file)
    elif file.name.endswith(".docx"):
        return parse_docx(file)
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    else:
        return ""

# ------------------------
# Upload JD & Resumes
# ------------------------
st.subheader("1) Upload Job Description")
jd_file = st.file_uploader("Upload JD (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"])

st.subheader("2) Upload Resumes")
resume_files = st.file_uploader("Upload one or more resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if jd_file and resume_files:
    jd_text = parse_file(jd_file)
    results = []

    # Parse resumes
    resumes_text = [parse_file(r) for r in resume_files]

    # TF-IDF vectorizer for hard match
    vectorizer = TfidfVectorizer().fit([jd_text] + resumes_text)

    for i, r_text in enumerate(resumes_text):
        # Hard match
        vectors = vectorizer.transform([jd_text, r_text])
        hard_score = cosine_similarity(vectors[0], vectors[1])[0][0]

        # Semantic match
        emb = model.encode([jd_text, r_text], convert_to_tensor=True)
        soft_score = util.pytorch_cos_sim(emb[0], emb[1]).item()

        # Weighted score
        final_score = 0.4 * hard_score + 0.6 * soft_score
        final_score = final_score * 100
        verdict = "High" if final_score >= 70 else "Medium" if final_score >= 50 else "Low"

        results.append({
            "Resume": resume_files[i].name,
            "Score": round(final_score, 2),
            "Verdict": verdict
        })

    df = pd.DataFrame(results).sort_values(by="Score", ascending=False).reset_index(drop=True)
    st.subheader("Results")
    st.dataframe(df, use_container_width=True)

    # Download CSV
    csv = df.to_csv(index=False).encode()
    st.download_button("Download Results CSV", data=csv, file_name="resume_scores.csv", mime="text/csv")
