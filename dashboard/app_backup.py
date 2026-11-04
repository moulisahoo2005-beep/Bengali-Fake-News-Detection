import streamlit as st
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Page configuration
st.set_page_config(
    page_title="Bengali Fake News Detection",
    page_icon="📰",
    layout="wide"
)

MODEL_PATH = "./BanglaBERT_Final_Model"

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    model.eval()
    return tokenizer, model


# Load threshold
with open(f"{MODEL_PATH}/best_threshold.json", "r") as file:
    threshold_data = json.load(file)

THRESHOLD = float(threshold_data["best_threshold"])


# Dashboard title
st.title("📰 Bengali Fake News Detection System")

st.write(
    "A Deep Learning Based Bengali Fake News Classification Dashboard"
)

st.divider()

st.header("About This Project")

st.write("""
This project uses BanglaBERT to classify Bengali news into:

- Fake News
- Authentic News
""")

st.info(
    "This model predicts categories based on learned patterns. "
    "It does not independently verify news authenticity."
)

st.divider()

# Load model
try:
    tokenizer, model = load_model()
    st.success("BanglaBERT Model Loaded Successfully!")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# News classification
st.header("News Classification")

news_text = st.text_area(
    "Enter Bengali News Text:",
    placeholder="এখানে বাংলা খবর লিখুন...",
    height=150
)

if st.button("Classify News", type="primary"):

    if not news_text.strip():
        st.warning("Please enter Bengali news text.")

    else:
        inputs = tokenizer(
            news_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(
                outputs.logits, dim=1
            )[0]

        fake_probability = probabilities[0].item()
        authentic_probability = probabilities[1].item()

        if fake_probability >= THRESHOLD:
            prediction = "Fake News"
            confidence = fake_probability
        else:
            prediction = "Authentic News"
            confidence = authentic_probability

        st.subheader("Prediction Result")

        if prediction == "Fake News":
            st.error(f"Prediction: {prediction}")
        else:
            st.success(f"Prediction: {prediction}")

        st.metric(
            "Prediction Confidence",
            f"{confidence * 100:.2f}%"
        )

        st.write("Fake News Probability")
        st.progress(fake_probability)

        st.write("Authentic News Probability")
        st.progress(authentic_probability)

        st.caption(f"Classification threshold: {THRESHOLD}")


st.divider()

st.subheader("Model Information")

st.write("Model: BanglaBERT")
st.write("Model Name: csebuetnlp/banglabert")
st.write("Maximum Sequence Length: 512")
st.write("Classes: Fake News and Authentic News")

st.caption(
    "Project: Bengali Fake News Detection | Powered by BanglaBERT"
)