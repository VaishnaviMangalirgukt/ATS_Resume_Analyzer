import os
import pandas as pd
import spacy
import PyPDF2
from tqdm import tqdm

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Get project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define paths
resumes_folder = os.path.join(project_root, "data", "resumes")
job_desc_csv = os.path.join(project_root, "data", "job_descriptions.csv")
processed_resumes_path = os.path.join(project_root, "data", "processed_resumes.csv")
processed_job_desc_path = os.path.join(project_root, "data", "processed_job_descriptions.csv")

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + " "
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text.strip()

# Process resumes
print("Processing new resumes...")
resume_data = []
for filename in tqdm(os.listdir(resumes_folder)):
    if filename.endswith(".pdf"):
        resume_path = os.path.join(resumes_folder, filename)
        resume_text = extract_text_from_pdf(resume_path)
        resume_data.append({"filename": filename, "text": resume_text})

# Convert to DataFrame and save
resumes_df = pd.DataFrame(resume_data)
if os.path.exists(processed_resumes_path):
    old_resumes_df = pd.read_csv(processed_resumes_path)
    resumes_df = pd.concat([old_resumes_df, resumes_df], ignore_index=True)
resumes_df.to_csv(processed_resumes_path, index=False)
print(f"Resumes updated: {processed_resumes_path}")

# Process job descriptions
print("Processing new job descriptions...")
if os.path.exists(job_desc_csv):
    jobs_df = pd.read_csv(job_desc_csv)
    jobs_df = jobs_df[["job_id", "job", "job_details"]]  # Keep only necessary columns
    if os.path.exists(processed_job_desc_path):
        old_jobs_df = pd.read_csv(processed_job_desc_path)
        jobs_df = pd.concat([old_jobs_df, jobs_df], ignore_index=True)
    jobs_df.to_csv(processed_job_desc_path, index=False)
    print(f"Job descriptions updated: {processed_job_desc_path}")
else:
    print("No new job descriptions found!")

print("Data update complete.")
