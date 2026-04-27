import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# =============================
# Page setup
# =============================
st.set_page_config(
    page_title="Fake News Detection AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================
# Custom CSS
# =============================
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0rem;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .card {
        padding: 1.1rem;
        border-radius: 18px;
        border: 1px solid rgba(120,120,120,0.18);
        background: rgba(250,250,250,0.04);
    }
    .good {color: #16a34a; font-weight: 800;}
    .bad {color: #dc2626; font-weight: 800;}
    .medium {color: #d97706; font-weight: 800;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# Constants
# =============================
MODEL_NAME = "laitali2026/fake-news-bert-model"
MAX_LEN = 512
DEFAULT_THRESHOLD = 0.50

MODEL_METRICS = {
    "Accuracy": 0.9638,
    "Precision": 0.9663,
    "Recall": 0.9575,
    "F1 Score": 0.9619,
}

TRAINING_HISTORY = pd.DataFrame(
    {
        "Epoch": [1, 2, 3],
        "Training Loss": [0.157617, 0.065196, 0.011758],
        "Validation Loss": [0.168930, 0.209829, 0.243374],
        "Accuracy": [0.957820, 0.959916, 0.961750],
        "Precision": [0.943796, 0.971942, 0.957228],
        "Recall": [0.970027, 0.943869, 0.963488],
        "F1": [0.956732, 0.957700, 0.960348],
    }
)

CONFUSION_MATRIX = pd.DataFrame(
    [[974, 12], [10, 1004]],
    index=["Actual Real", "Actual Fake"],
    columns=["Predicted Real", "Predicted Fake"],
)

# =============================
# Model loading
# =============================
@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def chunk_document(text: str, tokenizer, max_len: int = MAX_LEN) -> List[List[int]]:
    token_ids = tokenizer.encode(str(text), add_special_tokens=False, truncation=False)
    chunk_size = max_len - 2
    chunks = []

    for i in range(0, len(token_ids), chunk_size):
        chunk_tokens = token_ids[i : i + chunk_size]
        chunk_tokens = [tokenizer.cls_token_id] + chunk_tokens + [tokenizer.sep_token_id]
        chunks.append(chunk_tokens)

    if len(chunks) == 0:
        chunks = [[tokenizer.cls_token_id, tokenizer.sep_token_id]]

    return chunks


def predict_article(text: str, threshold: float) -> Dict:
    tokenizer, model = load_model()
    chunks = chunk_document(text, tokenizer)
    fake_probs = []

    for chunk in chunks:
        attention_mask = [1] * len(chunk)
        pad_length = MAX_LEN - len(chunk)

        input_ids = chunk + [tokenizer.pad_token_id] * pad_length
        attention_mask = attention_mask + [0] * pad_length

        inputs = {
            "input_ids": torch.tensor([input_ids]),
            "attention_mask": torch.tensor([attention_mask]),
        }

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            fake_probs.append(float(probs[1]))

    avg_fake_prob = float(np.mean(fake_probs))
    max_fake_prob = float(np.max(fake_probs))
    predicted_label = int(avg_fake_prob >= threshold)

    risk = assign_risk(avg_fake_prob)
    action = recommend_action(risk)

    return {
        "prediction": "Fake News" if predicted_label == 1 else "Real News",
        "predicted_label": predicted_label,
        "fake_probability": avg_fake_prob,
        "real_probability": 1 - avg_fake_prob,
        "max_fake_probability": max_fake_prob,
        "num_chunks": len(chunks),
        "chunk_fake_probs": fake_probs,
        "risk": risk,
        "action": action,
    }


def assign_risk(prob: float) -> str:
    if prob >= 0.75:
        return "High Risk"
    if prob >= 0.50:
        return "Medium Risk"
    return "Low Risk"


def recommend_action(risk: str) -> str:
    if risk == "High Risk":
        return "Immediate fact-checking recommended"
    if risk == "Medium Risk":
        return "Send to human reviewer"
    return "No immediate action needed"


def probability_gauge(fake_prob: float, threshold: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=fake_prob * 100,
            number={"suffix": "%"},
            title={"text": "Fake News Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#dc2626" if fake_prob >= threshold else "#16a34a"},
                "steps": [
                    {"range": [0, 50], "color": "rgba(22,163,74,0.25)"},
                    {"range": [50, 75], "color": "rgba(245,158,11,0.25)"},
                    {"range": [75, 100], "color": "rgba(220,38,38,0.25)"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(height=330, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def chunk_chart(chunk_probs: List[float]):
    df = pd.DataFrame(
        {
            "Chunk": list(range(1, len(chunk_probs) + 1)),
            "Fake Probability": chunk_probs,
        }
    )
    fig = px.bar(
        df,
        x="Chunk",
        y="Fake Probability",
        text=df["Fake Probability"].map(lambda x: f"{x:.2f}"),
        range_y=[0, 1],
        title="Chunk-Level Fake News Probability",
    )
    fig.update_traces(marker_color="#dc2626")
    fig.update_layout(height=360)
    return fig

# =============================
# Sidebar
# =============================
st.sidebar.title("🧠 Fake News AI")
st.sidebar.caption("BERT + chunk-based full-document analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Single Article Prediction",
        "Batch Prediction",
        "Model Dashboard",
        "Four Analytics",
        "About",
    ],
)

threshold = st.sidebar.slider(
    "Fake News Threshold",
    min_value=0.30,
    max_value=0.90,
    value=DEFAULT_THRESHOLD,
    step=0.05,
)

st.sidebar.markdown("---")
st.sidebar.write("**Model:**")
st.sidebar.code(MODEL_NAME)

# =============================
# Home
# =============================
if page == "Home":
    st.markdown('<div class="main-title">Fake News Detection AI Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">A Streamlit web app using a fine-tuned BERT model with chunk-based document processing.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{MODEL_METRICS['Accuracy']:.2%}")
    c2.metric("Precision", f"{MODEL_METRICS['Precision']:.2%}")
    c3.metric("Recall", f"{MODEL_METRICS['Recall']:.2%}")
    c4.metric("F1 Score", f"{MODEL_METRICS['F1 Score']:.2%}")

    st.markdown("---")

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Project Overview")
        st.write(
            "This application detects fake news articles using a fine-tuned BERT transformer model. "
            "Because BERT can only process 512 tokens at a time, long articles are split into chunks, "
            "classified chunk-by-chunk, and then aggregated into one final document-level prediction."
        )
        st.write(
            "The app supports real-time prediction, batch CSV prediction, model performance analytics, "
            "and prescriptive recommendations for content moderation."
        )

    with right:
        st.subheader("Pipeline")
        st.markdown(
            """
            1. Enter or upload article text  
            2. Tokenize using BERT tokenizer  
            3. Split long text into 512-token chunks  
            4. Predict each chunk  
            5. Average chunk probabilities  
            6. Assign risk level and action
            """
        )

# =============================
# Single Article Prediction
# =============================
elif page == "Single Article Prediction":
    st.title("Single Article Prediction")
    st.caption("Paste a full news article or headline + body text.")

    article_text = st.text_area("Article Text", height=300, placeholder="Paste article content here...")

    if st.button("Analyze Article", type="primary"):
        if not article_text.strip():
            st.warning("Please paste an article first.")
        else:
            with st.spinner("Loading BERT and analyzing article..."):
                result = predict_article(article_text, threshold)

            label_color = "bad" if result["predicted_label"] == 1 else "good"
            st.markdown(
                f"### Prediction: <span class='{label_color}'>{result['prediction']}</span>",
                unsafe_allow_html=True,
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Fake Probability", f"{result['fake_probability']:.2%}")
            m2.metric("Real Probability", f"{result['real_probability']:.2%}")
            m3.metric("Risk Level", result["risk"])
            m4.metric("Chunks Used", result["num_chunks"])

            st.plotly_chart(probability_gauge(result["fake_probability"], threshold), use_container_width=True)

            st.subheader("Recommended Action")
            if result["risk"] == "High Risk":
                st.error(result["action"])
            elif result["risk"] == "Medium Risk":
                st.warning(result["action"])
            else:
                st.success(result["action"])

            st.subheader("Chunk-Level Analysis")
            st.plotly_chart(chunk_chart(result["chunk_fake_probs"]), use_container_width=True)

            chunk_df = pd.DataFrame(
                {
                    "Chunk": list(range(1, len(result["chunk_fake_probs"]) + 1)),
                    "Fake Probability": result["chunk_fake_probs"],
                }
            )
            st.dataframe(chunk_df, use_container_width=True, hide_index=True)

# =============================
# Batch Prediction
# =============================
elif page == "Batch Prediction":
    st.title("Batch Prediction")
    st.caption("Upload a CSV file and predict many articles at once.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("CSV Preview")
        st.dataframe(data.head(), use_container_width=True)

        text_column = st.selectbox("Select text column", data.columns)

        max_rows = st.slider("Maximum rows to process", 1, min(100, len(data)), min(10, len(data)))

        if st.button("Run Batch Prediction", type="primary"):
            results = []
            progress = st.progress(0)

            for i, text in enumerate(data[text_column].fillna("").astype(str).head(max_rows)):
                pred = predict_article(text, threshold)
                results.append(
                    {
                        "prediction": pred["prediction"],
                        "fake_probability": pred["fake_probability"],
                        "risk": pred["risk"],
                        "recommended_action": pred["action"],
                        "num_chunks": pred["num_chunks"],
                    }
                )
                progress.progress((i + 1) / max_rows)

            output = pd.concat([data.head(max_rows).reset_index(drop=True), pd.DataFrame(results)], axis=1)
            st.success("Batch prediction completed.")
            st.dataframe(output, use_container_width=True)

            st.download_button(
                "Download Results CSV",
                output.to_csv(index=False).encode("utf-8"),
                file_name="fake_news_predictions.csv",
                mime="text/csv",
            )

# =============================
# Model Dashboard
# =============================
elif page == "Model Dashboard":
    st.title("Model Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{MODEL_METRICS['Accuracy']:.2%}")
    c2.metric("Precision", f"{MODEL_METRICS['Precision']:.2%}")
    c3.metric("Recall", f"{MODEL_METRICS['Recall']:.2%}")
    c4.metric("F1 Score", f"{MODEL_METRICS['F1 Score']:.2%}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(
            TRAINING_HISTORY,
            x="Epoch",
            y=["Training Loss", "Validation Loss"],
            markers=True,
            title="Training vs Validation Loss",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.line(
            TRAINING_HISTORY,
            x="Epoch",
            y=["Accuracy", "Precision", "Recall", "F1"],
            markers=True,
            title="Validation Metrics by Epoch",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Confusion Matrix")
    fig = px.imshow(
        CONFUSION_MATRIX,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Document-Level Confusion Matrix",
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Hyperparameter Summary")
    hp = pd.DataFrame(
        {
            "Experiment": ["2 Epochs", "3 Epochs"],
            "Batch Size": [8, 8],
            "Accuracy": [0.9890, 0.9638],
            "Precision": [0.9882, 0.9663],
            "Recall": [0.9901, 0.9575],
            "F1 Score": [0.9892, 0.9619],
        }
    )
    st.dataframe(hp, use_container_width=True, hide_index=True)

# =============================
# Four Analytics
# =============================
elif page == "Four Analytics":

    st.title("Four Analytics Framework")

    st.header("1. Descriptive Analytics — What happened?")

    st.write("""
    The WELFake dataset used in this project contains more than **72,000 labeled news articles** classified as real or fake news. 
    After cleaning the dataset and removing missing values, the final dataset used for training contained approximately **71,500 articles**.

    Each article was constructed by combining the **title and full article text**, allowing the model to analyze complete news content 
    rather than isolated headlines.

    The dataset was split using **stratified sampling** into:

    • Training set — 70%  
    • Validation set — 15%  
    • Test set — 15%

    After preprocessing and tokenization, the dataset expanded into chunk-level samples:

    • Training chunks: ~15,386  
    • Validation chunks: ~3,817  
    • Test chunks: ~3,896  

    The BERT model (bert-base-uncased) was fine-tuned on these chunks using the following hyperparameters:

    • Learning rate: 2e-5  
    • Batch size: 8  
    • Epochs tested: 2 and 3  
    • Weight decay: 0.01  

    Final evaluation results achieved approximately:

    • Accuracy: 96.38%  
    • Precision: 96.63%  
    • Recall: 95.75%  
    • F1 Score: 96.19%

    These results demonstrate strong performance in identifying linguistic patterns associated with misinformation.
    """)

    st.header("2. Diagnostic Analytics — Why did it happen?")

    st.write("""
    During dataset analysis, a key challenge emerged: **BERT has a maximum input length of 512 tokens**, while many real-world 
    news articles are significantly longer.

    When articles exceed this limit, BERT would normally truncate the text, potentially removing important contextual information.

    To address this limitation, the project implemented a **chunk-based document processing strategy**.

    The process works as follows:

    1. Articles are tokenized using the BERT tokenizer.
    2. Articles longer than 512 tokens are split into smaller segments (chunks).
    3. Each chunk is processed independently by the BERT classifier.
    4. Predictions from all chunks are aggregated to produce a final document-level probability.

    This approach ensures that **the entire article is analyzed**, rather than only the first portion of the text.

    Diagnostic evaluation through the confusion matrix showed strong performance:

    • 974 real articles correctly classified  
    • 1004 fake articles correctly classified  
    • 12 real articles misclassified as fake  
    • 10 fake articles misclassified as real  

    These results confirm that the chunking strategy successfully preserves contextual information needed for accurate predictions.
    """)

    st.header("3. Predictive Analytics — What is likely to happen?")

    st.write("""
    Predictive analytics focuses on estimating the likelihood that a new article contains misinformation.

    When a user submits an article through the application:

    1. The article text is tokenized using the BERT tokenizer.
    2. Long articles are divided into **512-token chunks**.
    3. Each chunk is analyzed by the fine-tuned BERT classifier.
    4. The model outputs probabilities for both classes:
       • Real news probability
       • Fake news probability
    5. Probabilities across all chunks are averaged to produce a **final document-level prediction**.

    For example, when testing a clearly misleading article about a miracle cure, the model produced:

    • Fake probability: 99.98%  
    • Real probability: 0.02%

    The system therefore classified the article as **Fake News** with extremely high confidence.

    This predictive capability allows the system to analyze new articles in real time and estimate the probability of misinformation.
    """)

    st.header("4. Prescriptive Analytics — What should be done?")

    st.write("""
    Prescriptive analytics converts predictions into **actionable recommendations**.

    The system uses a configurable **fake news probability threshold** (default = 0.50) to determine classification outcomes.

    Based on the predicted probability, the system assigns a risk category:

    • Low Risk (Fake probability < 0.50)  
      → Article likely legitimate, no action required.

    • Medium Risk (0.50 – 0.75)  
      → Article may require **human review**.

    • High Risk (> 0.75)  
      → Article likely misinformation, **immediate fact-checking recommended**.

    For example, the system flagged a test article claiming a miracle cure as **High Risk**, recommending immediate verification.

    This prescriptive layer transforms the predictive model into a **decision-support system** that can assist:

    • news organizations  
    • social media platforms  
    • fact-checking agencies  
    • content moderation teams

    in prioritizing which articles should be reviewed for misinformation.
    """)
# =============================
# About
# =============================
elif page == "About":
    st.title("About This Project")
    st.write(
        "This project was developed as a master's-level data science project for fake news detection using NLP and deep learning. "
        "It uses a fine-tuned BERT model trained on the WELFake dataset and deployed through Streamlit."
    )

    st.subheader("Technologies Used")
    st.markdown(
        """
        - Python
        - Streamlit
        - Hugging Face Transformers
        - PyTorch
        - Plotly
        - Pandas
        - Scikit-learn
        """
    )

    st.subheader("Model")
    st.code(MODEL_NAME)

    st.subheader("Deployment")
    st.write("The app is deployed using GitHub and Streamlit Community Cloud, while the trained model is hosted on Hugging Face Hub.")
