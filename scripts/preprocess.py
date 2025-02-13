import os
import pandas as pd
import pdfplumber
import re
import multiprocessing

def load_job_descriptions(csv_path):
    """Loads job descriptions from a configurable CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Selecting relevant columns
    df = df[['job', 'company_name', 'location', 'work_type', 'job_details']]
    
    # Cleaning text data
    df['job_details'] = df['job_details'].astype(str).apply(clean_text)
    return df

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF resume."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return clean_text(text)

def clean_text(text):
    """Cleans extracted text by removing special characters, extra spaces, and lowercasing."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def process_resume(resume_path):
    """Processes a single resume file."""
    text = extract_text_from_pdf(resume_path)
    return {'filename': os.path.basename(resume_path), 'text': text}

def process_resumes(resume_folder):
    """Processes all PDF resumes in the given folder using multiprocessing."""
    resume_files = [os.path.join(resume_folder, f) for f in os.listdir(resume_folder) if f.endswith(".pdf")]
    
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        resume_data = pool.map(process_resume, resume_files)
    
    return pd.DataFrame(resume_data)

if __name__ == "__main__":
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    job_csv = "linkedin_job_data.csv"  # Specify the exact file name
    job_csv_path = os.path.join(project_root, "data", "job_descriptions", job_csv)
    resume_folder = os.path.join(project_root, "data", "resumes")
    
    print(f"Looking for file at: {job_csv_path}")
    
    # Load job descriptions
    job_descriptions = load_job_descriptions(job_csv_path)
    print("Job descriptions loaded successfully.")
    
    # Save job descriptions
    job_descriptions.to_csv(os.path.join(project_root, "data", "processed_job_descriptions.csv"), index=False)
    print("Processed job descriptions saved.")
    
    # Process resumes
    print("Scanning resumes...")
    resumes = process_resumes(resume_folder)
    print("Resume processing complete.")
    
    # Save processed data
    resumes.to_csv(os.path.join(project_root, "data", "processed_resumes.csv"), index=False)
    print("Processed resumes saved.")
