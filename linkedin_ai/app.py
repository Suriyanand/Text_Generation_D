import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline

# Load pre-trained models
classifier = pipeline("sentiment-analysis")
generator = pipeline("text2text-generation", model="google/flan-t5-small")

# Title
st.title("🤖 LinkedIn Review Classifier & Responder")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file with a 'Review' column", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if 'Review' not in df.columns:
        st.error("❌ The file must contain a 'Review' column.")
    else:
        # Show the original data
        st.subheader("📋 Original Reviews")
        st.dataframe(df.head())

        # Sentiment classification
        st.subheader("🔍 Classifying Sentiment...")
        df["Sentiment"] = df["Review"].apply(lambda x: classifier(x)[0]['label'].lower())

        # Plot the sentiment distribution
        st.subheader("📊 Sentiment Distribution (Positive/Negative)")
        sentiment_counts = df["Sentiment"].value_counts()

        # Plot histogram
        plt.figure(figsize=(8, 6))
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="Set2")
        plt.title("Sentiment Distribution")
        plt.xlabel("Sentiment")
        plt.ylabel("Count of Reviews")
        st.pyplot(plt)

        # Text generation (responses)
        st.subheader("✍️ Generating Responses...")
        def generate_response(review):
            prompt = f"Write a professional response for customer: {review}"
            response = generator(prompt, max_length=50, do_sample=True)
            return response[0]["generated_text"]

        df["Response"] = df["Review"].apply(generate_response)

        # Show results
        st.subheader("✅ Final Output")
        st.dataframe(df)

        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results as CSV", csv, "linkedin_responses.csv", "text/csv")
