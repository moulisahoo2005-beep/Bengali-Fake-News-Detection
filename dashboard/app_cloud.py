
import streamlit as st
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# --------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Bengali Fake News Detection",
    page_icon="📰",
    layout="wide"
)

MODEL_ID = "moulisahoo/Bengali-Fake-News-BanglaBERT"
THRESHOLD = 0.485

# --------------------------------
# 2. LOAD MODEL FROM HUGGING FACE
# --------------------------------

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID
    )

    model.to("cpu")
    model.eval()

    return tokenizer, model


# --------------------------------
# 3. LOAD MODEL
# --------------------------------

st.title("📰 Bengali Fake News Detection")
st.write(
    "A Bengali Fake News Classification System "
    "using Fine-tuned BanglaBERT."
)

with st.spinner("Loading BanglaBERT model... Please wait."):

    try:
        tokenizer, model = load_model()
        st.success("BanglaBERT model loaded successfully!")

    except Exception as e:
        st.error("Model loading failed.")
        st.exception(e)
        st.stop()


# --------------------------------
# 4. SIDEBAR
# --------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a Section",
    [
        "News Classification",
        "Dataset Analysis",
        "Model Evaluation",
        "Model Information"
    ]
)


# --------------------------------
# 5. NEWS CLASSIFICATION
# --------------------------------

if page == "News Classification":

    st.header("🔍 Bengali News Classification")

    st.write(
        "Enter Bengali news text below to predict "
        "whether it is Fake News or Authentic News."
    )

    news_text = st.text_area(
        "Enter Bengali News:",
        height=180,
        placeholder="এখানে বাংলা সংবাদ লিখুন..."
    )

    if st.button("Predict News", type="primary"):

        if not news_text.strip():

            st.warning("Please enter some news text.")

        else:

            with st.spinner("Analyzing news..."):

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

                # Verified class mapping:
                # 0 = Fake News
                # 1 = Authentic News

                if fake_probability >= THRESHOLD:

                    prediction = "Fake News"
                    confidence = fake_probability

                else:

                    prediction = "Authentic News"
                    confidence = authentic_probability

            st.subheader("Prediction Result")

            if prediction == "Fake News":

                st.error("Prediction: Fake News")

            else:

                st.success("Prediction: Authentic News")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Fake News Probability",
                    f"{fake_probability * 100:.2f}%"
                )

            with col2:

                st.metric(
                    "Authentic News Probability",
                    f"{authentic_probability * 100:.2f}%"
                )

            st.metric(
                "Prediction Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.caption(
                f"Decision threshold: {THRESHOLD}"
            )


# --------------------------------
# 6. DATASET ANALYSIS
# --------------------------------

elif page == "Dataset Analysis":

    st.header("📊 Dataset Analysis")

    st.write(
        "Analysis based on the verified cleaned dataset "
        "used in this project."
    )

    # Verified counts from cleaned dataset

    class_counts = pd.Series(
        [9615, 46816],
        index=["Fake News", "Authentic News"]
    )

    split_counts = pd.Series(
        [45144, 5646, 5641],
        index=["Training", "Validation", "Testing"]
    )

    total_records = int(class_counts.sum())

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", f"{total_records:,}")
    col2.metric("Fake News", f"{class_counts['Fake News']:,}")
    col3.metric(
        "Authentic News",
        f"{class_counts['Authentic News']:,}"
    )

    st.subheader("Class Distribution")

    fig, ax = plt.subplots()

    class_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("News Category")
    ax.set_ylabel("Number of Records")
    ax.set_title("Dataset Class Distribution")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Class Distribution Percentage")

    fig, ax = plt.subplots()

    class_counts.plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=ax
    )

    ax.set_ylabel("")
    ax.set_title("Fake vs Authentic News")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Dataset Split Distribution")

    fig, ax = plt.subplots()

    split_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Dataset Split")
    ax.set_ylabel("Number of Records")
    ax.set_title("Training, Validation and Testing")

    st.pyplot(fig)
    plt.close(fig)


# --------------------------------
# 7. MODEL EVALUATION
# --------------------------------

elif page == "Model Evaluation":

    st.header("📈 BanglaBERT Model Evaluation")

    st.write(
        "Final evaluation results on the held-out test dataset."
    )

    metrics = {
        "Accuracy": "97.00%",
        "Macro Precision": "96.97%",
        "Macro Recall": "92.30%",
        "Macro F1-Score": "94.44%",
        "Balanced Accuracy": "92.30%"
    }

    cols = st.columns(3)

    for i, (name, value) in enumerate(metrics.items()):

        with cols[i % 3]:

            st.metric(name, value)

    st.subheader("Confusion Matrix")

    confusion_matrix = [
        [820, 143],
        [26, 4652]
    ]

    fig, ax = plt.subplots()

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
    ax.set_title("BanglaBERT Test Confusion Matrix")

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Rows represent actual labels; columns represent predicted labels."
    )

    st.subheader("Model Comparison")

    comparison = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1-Score"
        ],
        "TF-IDF + Logistic Regression": [
            94.42, 92.86, 86.70, 89.39
        ],
        "BanglaBERT": [
            97.00, 96.97, 92.30, 94.44
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )


# --------------------------------
# 8. MODEL INFORMATION
# --------------------------------

elif page == "Model Information":

    st.header("🤖 Model Information")

    st.write("**Model Name:** BanglaBERT")

    st.write(
        "**Hugging Face Model:** "
        "csebuetnlp/banglabert"
    )

    st.write(
        "**Model Repository:** "
        "moulisahoo/Bengali-Fake-News-BanglaBERT"
    )

    st.write("**Architecture:** ELECTRA-based BanglaBERT")

    st.write("**Number of Classes:** 2")

    st.write("**Maximum Token Length:** 512")

    st.write("**Decision Threshold:** 0.485")

    st.write("**Class 0:** Fake News")

    st.write("**Class 1:** Authentic News")

    st.write("**Framework:** PyTorch and Hugging Face Transformers")


# --------------------------------
# 9. DISCLAIMER
# --------------------------------

st.divider()

st.warning(
    "Disclaimer: This system predicts news categories based on "
    "patterns learned from the training dataset. It does not "
    "independently verify whether a news story is factually true. "
    "Predictions may be incorrect."
)

st.caption(
    "Bengali Fake News Detection Project | BanglaBERT"
)
