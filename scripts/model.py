import os
import pandas as pd
import numpy as np
import spacy
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import wordnet
from tqdm import tqdm

# Load NLP model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Get project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define file paths
processed_resumes_path = os.path.join(project_root, "data", "processed_resumes.csv")
processed_job_desc_path = os.path.join(project_root, "data", "processed_job_descriptions.csv")

# Load processed data
print("Loading data...")
resumes_df = pd.read_csv(processed_resumes_path)
jobs_df = pd.read_csv(processed_job_desc_path)
print("Data loaded successfully.")

# Handle missing values
resumes_df['text'] = resumes_df['text'].fillna('')
jobs_df['job_details'] = jobs_df['job_details'].fillna('')

# Text Preprocessing: Synonyms Replacement
def replace_synonyms(text):
    words = text.split()
    new_words = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            new_words.append(synonyms[0].lemmas()[0].name())  # Replace with first synonym
        else:
            new_words.append(word)
    return " ".join(new_words)

resumes_df['text'] = resumes_df['text'].apply(replace_synonyms)
jobs_df['job_details'] = jobs_df['job_details'].apply(replace_synonyms)

# Extract Entities (Skills, Degrees, etc.)
def extract_entities(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PERSON", "NORP"]]
    return " ".join(entities)

resumes_df['entities'] = resumes_df['text'].apply(extract_entities)
jobs_df['entities'] = jobs_df['job_details'].apply(extract_entities)

# Load optimized BERT model with GPU support
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Loading BERT model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

# Generate BERT embeddings with batch processing
print("Generating BERT embeddings...")
resume_embeddings = model.encode(resumes_df['text'].tolist(), batch_size=32, convert_to_tensor=True)
job_embeddings = model.encode(jobs_df['job_details'].tolist(), batch_size=32, convert_to_tensor=True)
print("Embeddings generated successfully.")

# Compute similarity scores using BERT (Dot Product instead of Cosine Similarity)
print("Computing BERT similarity scores...")
bert_similarity = np.dot(resume_embeddings.cpu().numpy(), job_embeddings.cpu().numpy().T)

# Compute TF-IDF similarity
print("Computing TF-IDF similarity scores...")
tfidf_vectorizer = TfidfVectorizer()
tfidf_resume_matrix = tfidf_vectorizer.fit_transform(resumes_df['text'])
tfidf_job_matrix = tfidf_vectorizer.transform(jobs_df['job_details'])
tfidf_similarity = cosine_similarity(tfidf_resume_matrix, tfidf_job_matrix)

# Normalize similarity scores
scaler = MinMaxScaler()
bert_similarity = scaler.fit_transform(bert_similarity)
tfidf_similarity = scaler.fit_transform(tfidf_similarity)

# Combine BERT and TF-IDF scores (Adjusted Weighting: 90% BERT, 10% TF-IDF)
final_similarity = (bert_similarity * 0.9) + (tfidf_similarity * 0.1)
print("Similarity computation complete.")

def get_top_matches(similarity_matrix, resumes_df, jobs_df, top_n=3):
    top_matches = []
    mean_similarity_scores = []
    
    num_resumes = similarity_matrix.shape[0]  # Get the number of resumes
    for i in range(num_resumes):
        top_indices = np.argsort(similarity_matrix[i])[::-1][:top_n]
        matched_jobs = [(jobs_df.iloc[j]['job'], similarity_matrix[i, j]) for j in top_indices]
        mean_similarity_scores.append(np.mean([similarity_matrix[i, j] for j in top_indices]))
        top_matches.append({"resume_index": i, "matched_jobs": matched_jobs})
    
    avg_similarity = np.mean(mean_similarity_scores)  # Compute mean similarity score
    print(f"Mean Similarity Score: {avg_similarity:.4f}")
    return top_matches, avg_similarity

# Find top job matches for resumes
print("Finding top job matches for resumes...")
top_matches, avg_similarity = get_top_matches(final_similarity, resumes_df, jobs_df, top_n=3)

# Save results
results_path = os.path.join(project_root, "data", "matching_results.csv")
results_df = pd.DataFrame([{**{"resume_index": match["resume_index"]}, **{"job_{}_match_score".format(idx+1): job for idx, job in enumerate(match["matched_jobs"])} } for match in top_matches])
results_df.to_csv(results_path, index=False)
print("Matching results saved to", results_path)

# Save similarity score for reference
eval_results_path = os.path.join(project_root, "data", "evaluation_results.txt")
with open(eval_results_path, "w") as f:
    f.write(f"Mean Similarity Score: {avg_similarity:.4f}\n")
print("Evaluation results saved to", eval_results_path)
