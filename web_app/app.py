import os
import fitz  # PyMuPDF for PDF text extraction
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def load_text(file_path):
    """Extract text from PDFs or TXT files."""
    try:
        if file_path.endswith(".pdf"):
            text = ""
            with fitz.open(file_path) as pdf:
                for page in pdf:
                    text += page.get_text("text")  # Extract text from each page
            return text.strip()
        elif file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read().strip()
        else:
            return ""
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def rank_resumes(job_desc_path, resume_files):
    """Rank only the newly uploaded resumes based on similarity score."""
    try:
        job_desc_text = load_text(job_desc_path)
        if not job_desc_text:
            return [{"error": "Job description is empty or unreadable."}]

        job_desc_embedding = model.encode(job_desc_text, convert_to_tensor=True)
        
        results = []
        for resume_file in resume_files:
            resume_path = os.path.join(UPLOAD_FOLDER, secure_filename(resume_file.filename))
            resume_file.save(resume_path)  # Save uploaded file

            resume_text = load_text(resume_path)  # Extract text from PDF or TXT
            if not resume_text:
                print(f"Warning: No text extracted from {resume_file.filename}")
                continue

            resume_embedding = model.encode(resume_text, convert_to_tensor=True)

            similarity = util.pytorch_cos_sim(job_desc_embedding, resume_embedding).item() * 100
            
            results.append({
                "resume": resume_file.filename,
                "similarity": round(similarity, 2)
            })

        results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        return results

    except Exception as e:
        print(f"Error ranking resumes: {e}")
        return [{"error": "An error occurred while ranking resumes."}]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "job_description" not in request.files or "resumes" not in request.files:
        return jsonify({"error": "Missing job description or resumes."}), 400

    job_desc_file = request.files["job_description"]
    resumes_files = request.files.getlist("resumes")

    if job_desc_file.filename == "" or len(resumes_files) == 0:
        return jsonify({"error": "No selected file(s)."}), 400

    job_desc_path = os.path.join(UPLOAD_FOLDER, secure_filename(job_desc_file.filename))
    job_desc_file.save(job_desc_path)

    results = rank_resumes(job_desc_path, resumes_files)

    return jsonify({"success": True, "results": results})

import os
import fitz  # PyMuPDF for PDF text extraction
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def load_text(file_path):
    """Extract text from PDFs or TXT files."""
    try:
        if file_path.endswith(".pdf"):
            text = ""
            with fitz.open(file_path) as pdf:
                for page in pdf:
                    text += page.get_text("text")  # Extract text from each page
            return text.strip()
        elif file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read().strip()
        else:
            return ""
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def rank_resumes(job_desc_path, resume_files):
    """Rank only the newly uploaded resumes based on similarity score."""
    try:
        job_desc_text = load_text(job_desc_path)
        if not job_desc_text:
            return [{"error": "Job description is empty or unreadable."}]

        job_desc_embedding = model.encode(job_desc_text, convert_to_tensor=True)
        
        results = []
        for resume_file in resume_files:
            resume_path = os.path.join(UPLOAD_FOLDER, secure_filename(resume_file.filename))
            resume_file.save(resume_path)  # Save uploaded file

            resume_text = load_text(resume_path)  # Extract text from PDF or TXT
            if not resume_text:
                print(f"Warning: No text extracted from {resume_file.filename}")
                continue

            resume_embedding = model.encode(resume_text, convert_to_tensor=True)

            similarity = util.pytorch_cos_sim(job_desc_embedding, resume_embedding).item() * 100
            
            results.append({
                "resume": resume_file.filename,
                "similarity": round(similarity, 2)
            })

        results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        return results

    except Exception as e:
        print(f"Error ranking resumes: {e}")
        return [{"error": "An error occurred while ranking resumes."}]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "job_description" not in request.files or "resumes" not in request.files:
        return jsonify({"error": "Missing job description or resumes."}), 400

    job_desc_file = request.files["job_description"]
    resumes_files = request.files.getlist("resumes")

    if job_desc_file.filename == "" or len(resumes_files) == 0:
        return jsonify({"error": "No selected file(s)."}), 400

    job_desc_path = os.path.join(UPLOAD_FOLDER, secure_filename(job_desc_file.filename))
    job_desc_file.save(job_desc_path)

    results = rank_resumes(job_desc_path, resumes_files)

    return jsonify({"success": True, "results": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501)  # or 9000 / 5050



