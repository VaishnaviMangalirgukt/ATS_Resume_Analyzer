import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

# Get project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define file paths
processed_resumes_path = os.path.join(project_root, "data", "processed_resumes.csv")
processed_job_desc_path = os.path.join(project_root, "data", "processed_job_descriptions.csv")

# Load processed data
resumes = pd.read_csv(processed_resumes_path)
job_descriptions = pd.read_csv(processed_job_desc_path)
print("Data loaded successfully.")

# Handle missing values
resumes.dropna(subset=['text'], inplace=True)  # Remove rows where 'text' is NaN
resumes['text'] = resumes['text'].fillna('')  # Replace any remaining NaN values with an empty string

job_descriptions.dropna(subset=['job_details'], inplace=True)
job_descriptions['job_details'] = job_descriptions['job_details'].fillna('')

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Vectorize resumes
resume_tfidf = tfidf_vectorizer.fit_transform(resumes['text'])
print("Resume text vectorized successfully.")

# Vectorize job descriptions
job_tfidf = tfidf_vectorizer.transform(job_descriptions['job_details'])
print("Job descriptions vectorized successfully.")

# Save vectorized data
save_npz(os.path.join(project_root, "data", "resume_tfidf.npz"), resume_tfidf)
save_npz(os.path.join(project_root, "data", "job_tfidf.npz"), job_tfidf)

# Save TF-IDF matrices using joblib
joblib.dump(resume_tfidf, os.path.join(project_root, "data", "tfidf_resumes.pkl"))
joblib.dump(job_tfidf, os.path.join(project_root, "data", "tfidf_jobs.pkl"))

print("Vectorization complete. TF-IDF matrices saved successfully.")
