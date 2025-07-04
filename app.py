import streamlit as st
from transformers import pipeline
import torch

# Page configuration
st.set_page_config(page_title="📝 Text Summarizer", layout="centered")

# App title
st.title("🧠 AI-Powered Text Summarizer")
st.markdown("Enter a paragraph or article below, and get a short, meaningful summary using **T5 Transformer**.")

# Input text area
with st.container():
    text = st.text_area("📄 Enter your text to summarize", height=200)

# Sidebar settings
st.sidebar.header("⚙️ Summary Settings")
max_length = st.sidebar.slider("Maximum Summary Length", min_value=20, max_value=300, value=100, step=10)
min_length = st.sidebar.slider("Minimum Summary Length", min_value=10, max_value=max_length-10, value=30, step=5)

# Process when text is entered
if text:
    with st.spinner("🔍 Generating summary..."):
        summarizer = pipeline(
            "summarization",
            model="t5-small",
            tokenizer="t5-small",
            framework="pt",
            do_sample=False,
            max_length=max_length,
            min_length=min_length,
        )
        summary = summarizer("summarize: " + text)
        st.success("✅ Summary generated successfully!")

        # Output
        st.markdown("### ✂️ Summary:")
        st.code(summary[0]['summary_text'], language='text')
