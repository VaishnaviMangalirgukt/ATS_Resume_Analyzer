import os
import base64
import fitz  # PyMuPDF for PDF text extraction
import streamlit as st
from sentence_transformers import SentenceTransformer, util

# ✅ Set Streamlit Page Config FIRST
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# Load NLP model for similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Gets the directory of app.py
IMAGE_PATH = os.path.join(BASE_DIR, "static", "background1.AVIF")

# Function to encode image to base64
def get_base64_encoded_image(image_path):
    """Convert image to base64 for inline CSS background."""
    if not os.path.exists(image_path):
        st.error(f"❌ Error: Background image not found at {image_path}")
        return ""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Get encoded background image
image_base64 = get_base64_encoded_image(IMAGE_PATH)

# ✅ Apply Background Image & Styles
st.markdown(
    f"""
    <style>
        .stApp {{
            background: url("data:image/avif;base64,{image_base64}") no-repeat center center fixed;
            background-size: cover;
        }}
        h1 {{
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: white;
        }}
        h4 {{
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            color: black;
        }}
        .upload-box {{
            border: 2px dashed #4CAF50;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
            background-color: white;
            margin: auto;
            width: 20%;  /* ✅ Reduced width */
        }}
        .stButton>button {{
            display: block;
            margin: auto;
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border-radius: 8px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ UI Layout
st.markdown("<h1>AI Resume Analyzer</h1>", unsafe_allow_html=True)
st.markdown(
    "<h4>Upload a job description and resumes to analyze their similarity.</h4>",
    unsafe_allow_html=True
)

# ✅ Function to extract text from uploaded files
def load_text(uploaded_file):
    """Extract text from an uploaded file (PDF or TXT)."""
    try:
        if uploaded_file.type == "application/pdf":
            text = ""
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as pdf:
                for page in pdf:
                    text += page.get_text("text")
            return text.strip()
        
        elif uploaded_file.type == "text/plain":
            return uploaded_file.read().decode("utf-8").strip()
        
        else:
            return ""
    
    except Exception as e:
        st.error(f"Error reading {uploaded_file.name}: {e}")
        return ""

# ✅ Function to rank resumes based on job description
def rank_resumes(job_desc_text, resume_files):
    """Compute similarity scores and rank resumes."""
    try:
        if not job_desc_text:
            return [{"error": "Job description is empty or unreadable."}]

        job_desc_embedding = model.encode(job_desc_text, convert_to_tensor=True)

        results = []
        for resume_file in resume_files:
            resume_text = load_text(resume_file)
            if not resume_text:
                continue  # Skip empty files

            resume_embedding = model.encode(resume_text, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(job_desc_embedding, resume_embedding).item() * 100

            results.append({
                "resume": resume_file.name,
                "similarity": round(similarity, 2)
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    except Exception as e:
        st.error(f"Error ranking resumes: {e}")
        return [{"error": "An error occurred while ranking resumes."}]

# ✅ Upload job description (smaller box)
st.markdown('<div class="upload-box">📂 <b>Upload Job Description (PDF/TXT)</b></div>', unsafe_allow_html=True)
job_desc_file = st.file_uploader("", type=["pdf", "txt"], key="job_desc", label_visibility="collapsed")

# ✅ Upload multiple resumes (smaller box)
st.markdown('<div class="upload-box">📂 <b>Upload Resumes (PDF/TXT)</b></div>', unsafe_allow_html=True)
resume_files = st.file_uploader("", type=["pdf", "txt"], accept_multiple_files=True, key="resumes", label_visibility="collapsed")

# ✅ Centered Analyze Button
if st.button("Analyze"):
    if job_desc_file and resume_files:
        job_desc_text = load_text(job_desc_file)

        results = rank_resumes(job_desc_text, resume_files)

        st.subheader("🔍 Ranking Results:")
        for res in results:
            if "error" in res:
                st.warning(res["error"])
            else:
                st.write(f"**{res['resume']}**: {res['similarity']}% match")
    else:
        st.warning("Please upload both a job description and resumes to analyze.")
