import streamlit as st
import pandas as pd
import fasttext
import os
import re

MODEL_PATH = "model.bin"
DATA_PATH = "Dataset_CCE2.xlsx - train.csv"
TRAIN_TXT_PATH = "train_fasttext.txt"

import string

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Basic cleaning
    text = re.sub(r'\n', ' ', text)
    # Remove Hindi and English punctuation so FastText tokenizes words correctly
    hindi_punc = "।!?,.'\"()-" + string.punctuation
    for p in hindi_punc:
        text = text.replace(p, ' ')
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

@st.cache_resource
def load_or_train_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Training FastText model for the first time... This will take a few seconds."):
            # Read CSV
            try:
                df = pd.read_csv(DATA_PATH)
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
                st.stop()
            
            # Drop nulls
            df = df.dropna(subset=['text', 'experience'])
            
            # FastText format: __label__<label> <text>
            with open(TRAIN_TXT_PATH, "w", encoding="utf-8") as f:
                for idx, row in df.iterrows():
                    try:
                        label = str(int(row['experience']))
                    except:
                        continue
                    text = clean_text(row['text'])
                    f.write(f"__label__{label} {text}\n")
            
            # Train the model with limited buckets and dimensions to keep it under 10MB!
            model = fasttext.train_supervised(input=TRAIN_TXT_PATH, epoch=25, wordNgrams=2, bucket=20000, dim=50)
            model.save_model(MODEL_PATH)
            
            # Cleanup text file to save space
            if os.path.exists(TRAIN_TXT_PATH):
                os.remove(TRAIN_TXT_PATH)
    
    return fasttext.load_model(MODEL_PATH)

def main():
    st.set_page_config(page_title="Hindi Movie Sentiment Chatbot", page_icon="🎬")
    st.title("🎬 Hindi Movie Sentiment Chatbot")
    st.write("Type a Hindi movie review below to see if the sentiment is Positive, Neutral, or Negative!  \n*(Uses Facebook's FastText Model)*")

    # Load/Train the model
    model = load_or_train_model()

    # Chatbot UI
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Enter a Hindi review (e.g. 'यह फिल्म बहुत अच्छी है')...")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Predict using FastText
        cleaned_input = clean_text(user_input)
        
        # fasttext predict returns labels and probabilities
        labels, probabilities = model.predict(cleaned_input)
        
        # Parse label
        raw_label = labels[0].replace('__label__', '')
        
        # Map label to sentiment
        sentiment_map = {
            "0": "😞 Negative (नकारात्मक)",
            "1": "😐 Neutral (तटस्थ)",
            "2": "😃 Positive (सकारात्मक)"
        }
        
        sentiment = sentiment_map.get(raw_label, "Unknown Label")
        confidence = probabilities[0] * 100
        
        response = f"**Sentiment:** {sentiment} \n\n**Confidence:** {confidence:.2f}%"
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()
