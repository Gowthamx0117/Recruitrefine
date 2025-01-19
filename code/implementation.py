import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download("stopwords")
nltk.download("punkt")

# Preprocessing function
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphanumeric characters
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens)

# Streamlit app
def main():
    st.title("AI-Powered Resume Screening")
    st.write("Upload a dataset of resumes and job descriptions to calculate similarity scores.")

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        # Load dataset
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.write(df.head())

        # Ensure required columns are present
        if "Resume" not in df.columns or "Job_Description" not in df.columns:
            st.error("The dataset must contain 'Resume' and 'Job_Description' columns.")
            return

        # Preprocess the text
        st.write("Preprocessing text...")
        df["Resume_Cleaned"] = df["Resume"].apply(preprocess_text)
        df["Job_Description_Cleaned"] = df["Job_Description"].apply(preprocess_text)

        # Vectorize using TF-IDF
        st.write("Calculating similarity scores...")
        tfidf = TfidfVectorizer()
        combined_text = df["Resume_Cleaned"].tolist() + df["Job_Description_Cleaned"].tolist()
        vectors = tfidf.fit_transform(combined_text)
        resume_vectors = vectors[:len(df)]
        job_description_vectors = vectors[len(df):]

        # Calculate similarity
        similarities = cosine_similarity(resume_vectors, job_description_vectors)
        df["Similarity_Score"] = similarities.diagonal()

        # Sort by similarity score
        ranked_df = df.sort_values(by="Similarity_Score", ascending=False)
        st.write("Ranked Resumes:")
        st.write(ranked_df[["Resume", "Job_Description", "Similarity_Score"]])

        # Option to download ranked results
        csv_data = ranked_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Ranked Resumes",
            data=csv_data,
            file_name="ranked_resumes.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()
