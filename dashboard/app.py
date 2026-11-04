
import streamlit as st
import torch
import json
import pandas as pd
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Bengali Fake News Detection",
    page_icon="📰",
    layout="wide"
)

MODEL_PATH = "./BanglaBERT_Final_Model"

# ---------------- LOAD MODEL ----------------

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


with open(f"{MODEL_PATH}/best_threshold.json", "r") as file:
    threshold_data = json.load(file)

THRESHOLD = float(threshold_data["best_threshold"])

# ---------------- LOAD DATASET STATISTICS ----------------

@st.cache_data
def load_dataset_stats():

    train_labels = pd.read_csv(
        "data/processed/train.csv",
        usecols=["Label"]
    )["Label"]

    validation_labels = pd.read_csv(
        "data/processed/validation.csv",
        usecols=["Label"]
    )["Label"]

    test_labels = pd.read_csv(
        "data/processed/test.csv",
        usecols=["Label"]
    )["Label"]

    all_labels = pd.concat(
        [train_labels, validation_labels, test_labels],
        ignore_index=True
    )

    all_labels = pd.to_numeric(
        all_labels,
        errors="coerce"
    ).dropna().astype(int)

    class_counts = all_labels.value_counts().reindex(
        [0, 1],
        fill_value=0
    )

    split_counts = pd.DataFrame({
        "Records": [
            len(train_labels),
            len(validation_labels),
            len(test_labels)
        ]
    }, index=["Train", "Validation", "Test"])

    return class_counts, split_counts


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("📰 BanglaBERT")
    st.write("Fake News Detection")

    st.divider()

    st.subheader("Navigation")
    st.markdown("""
    - 🏠 Home
    - 🔍 News Classification
    - 📊 Dataset Analysis
    - 🤖 Model Information
    """)

    st.divider()
    st.caption("Bengali NLP Deep Learning Project")


# ---------------- MAIN DASHBOARD ----------------

st.title("📰 Bengali Fake News Detection")

st.markdown(
    "### AI-powered Bengali News Classification Dashboard"
)

st.write(
    "Detect potential Fake News and Authentic News "
    "using a fine-tuned BanglaBERT model."
)

st.divider()


# ---------------- LOAD MODEL ----------------

try:
    tokenizer, model = load_model()
    st.success("✅ BanglaBERT Model Loaded Successfully!")

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# ---------------- NEWS CLASSIFICATION ----------------

st.header("🔍 News Classification")

news_text = st.text_area(
    "Enter Bengali News Text",
    placeholder="এখানে বাংলা খবর লিখুন...",
    height=180
)

if st.button("🔎 Classify News", type="primary"):

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
                outputs.logits,
                dim=1
            )[0]

        fake_probability = probabilities[0].item()
        authentic_probability = probabilities[1].item()

        if fake_probability >= THRESHOLD:
            prediction = "Fake News"
            confidence = fake_probability
        else:
            prediction = "Authentic News"
            confidence = authentic_probability

        st.divider()
        st.subheader("📊 Prediction Result")

        if prediction == "Fake News":
            st.error(f"Prediction: {prediction}")
        else:
            st.success(f"Prediction: {prediction}")

        st.metric(
            "Prediction Confidence",
            f"{confidence * 100:.2f}%"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Fake News Probability**")
            st.progress(fake_probability)
            st.write(f"{fake_probability * 100:.2f}%")

        with col2:
            st.write("**Authentic News Probability**")
            st.progress(authentic_probability)
            st.write(f"{authentic_probability * 100:.2f}%")

        st.caption(
            f"Classification Threshold: {THRESHOLD}"
        )


# ---------------- DATASET ANALYSIS ----------------

st.divider()
st.header("📊 Dataset Analysis")

try:
    class_counts, split_counts = load_dataset_stats()

    fake_count = int(class_counts.loc[0])
    authentic_count = int(class_counts.loc[1])

    total_records = fake_count + authentic_count

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", f"{total_records:,}")

    with col2:
        st.metric("Fake News Records", f"{fake_count:,}")

    with col3:
        st.metric("Authentic News Records", f"{authentic_count:,}")

    st.subheader("News Class Distribution")

    chart_data = pd.DataFrame({
        "Number of Records": [
            fake_count,
            authentic_count
        ]
    }, index=["Fake News", "Authentic News"])

    st.bar_chart(chart_data)

    st.subheader("Class Distribution Pie Chart")

    fig, ax = plt.subplots()

    ax.pie(
        [fake_count, authentic_count],
        labels=["Fake News", "Authentic News"],
        autopct="%1.1f%%",
        startangle=90
    )

    ax.axis("equal")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Train, Validation and Test Distribution")

    st.bar_chart(split_counts)

    st.dataframe(
        split_counts,
        use_container_width=True
    )

except Exception as e:
    st.error(f"Dataset analysis failed: {e}")


# ---------------- MODEL INFORMATION ----------------

st.divider()
st.header("🤖 Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "BanglaBERT")

with col2:
    st.metric("Maximum Length", "512")

with col3:
    st.metric("Number of Classes", "2")

st.write("**Model Name:** `csebuetnlp/banglabert`")

st.write("**Classes:** Fake News | Authentic News")

st.write(f"**Classification Threshold:** {THRESHOLD}")

# ==========================================
# MODEL EVALUATION
# ==========================================

st.markdown("---")
st.header("📈 Model Evaluation")

st.write(
    "Performance of BanglaBERT on the independent test dataset."
)

# Evaluation Metrics
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Accuracy", "97.00%")
col2.metric("Precision", "96.97%")
col3.metric("Recall", "92.30%")
col4.metric("Macro F1", "94.44%")
col5.metric("Balanced Accuracy", "92.30%")

# Confusion Matrix
st.subheader("Confusion Matrix")

confusion_matrix = [
    [820, 143],
    [26, 4652]
]

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(6, 4))

sns.heatmap(
    confusion_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Fake News", "Authentic News"],
    yticklabels=["Fake News", "Authentic News"],
    ax=ax
)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")
ax.set_title("BanglaBERT Confusion Matrix")

st.pyplot(fig)
plt.close(fig)

st.caption(
    "Results are from the independent test dataset "
    "using the validation-selected threshold of 0.485."
)


# ---------------- DISCLAIMER ----------------

st.divider()

st.warning(
    "Disclaimer: This model predicts news categories based "
    "on learned patterns. It does not independently verify "
    "whether a news story is factually true or false."
)

st.caption(
    "Bengali Fake News Detection Project | Powered by BanglaBERT"
)

# ==========================================
# MODEL COMPARISON
# ==========================================

st.markdown("---")
st.header("⚖️ Model Comparison")

st.write(
    "Comparison between TF-IDF + Logistic Regression "
    "and BanglaBERT on the independent test dataset."
)

comparison_data = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1"
    ],
    "TF-IDF": [
        94.42,
        92.86,
        86.70,
        89.39
    ],
    "BanglaBERT": [
        97.00,
        96.97,
        92.30,
        94.44
    ]
})

st.subheader("Performance Comparison")

st.dataframe(
    comparison_data,
    use_container_width=True,
    hide_index=True
)

st.subheader("Model Performance Chart")

chart_df = comparison_data.set_index("Metric")

st.bar_chart(chart_df)

st.success(
    "BanglaBERT achieved a higher Macro F1-score "
    "than the TF-IDF baseline on the test dataset."
)

st.caption(
    "Both models were evaluated on the same independent "
    "test dataset. Results are from previously completed experiments."
)
