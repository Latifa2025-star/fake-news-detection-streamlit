import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

# Optional ML dependencies
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False


# -----------------------------
# App Configuration
# -----------------------------
st.set_page_config(
    page_title="Fake News Analytics Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
        .main {padding-top: 1rem;}
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
        .metric-card {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            padding: 1rem;
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .section-card {
            background: #111827;
            border-radius: 18px;
            padding: 1rem 1.25rem;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .small-muted {color: #9ca3af; font-size: 0.95rem;}
        .stAlert {border-radius: 14px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Constants / Defaults
# -----------------------------
MODEL_DIR = os.getenv("MODEL_DIR", "./bert_fake_news")
MAX_LEN = 512
CHUNK_TEXT_TOKENS = 510  # reserve [CLS] and [SEP]
DEFAULT_THRESHOLD = 0.50


@dataclass
class AppMetrics:
    accuracy: float = 0.9638
    precision: float = 0.9663
    recall: float = 0.9575
    f1: float = 0.9619
    eval_loss: float = 0.2455


DEFAULT_METRICS = AppMetrics()


# -----------------------------
# Helper Data
# -----------------------------
def get_training_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "training_loss": [0.157617, 0.065196, 0.011758],
            "validation_loss": [0.168930, 0.209829, 0.243374],
            "accuracy": [0.957820, 0.959916, 0.961750],
            "precision": [0.943796, 0.971942, 0.957228],
            "recall": [0.970027, 0.943869, 0.963488],
            "f1": [0.956732, 0.957700, 0.960348],
        }
    )


def get_confusion_matrix_df() -> pd.DataFrame:
    # From notebook example shown earlier
    cm = np.array([[974, 12], [10, 1004]])
    return pd.DataFrame(cm, index=["Actual Real", "Actual Fake"], columns=["Pred Real", "Pred Fake"])


def get_sample_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "headline": [
                "Authorities deny viral claim about election machines",
                "Scientists publish peer-reviewed climate update",
                "Celebrity shock story spreads without source links",
                "Government budget report released with official PDF",
            ],
            "predicted_label": ["Real", "Real", "Fake", "Real"],
            "fake_probability": [0.08, 0.03, 0.91, 0.12],
            "risk_category": ["Low", "Low", "High", "Low"],
            "action": ["No action", "No action", "Immediate fact-check", "No action"],
        }
    )


# -----------------------------
# Model Utilities
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model_bundle(model_dir: str):
    if not TRANSFORMERS_AVAILABLE:
        return None, None, "Transformers not installed in this environment."

    if not os.path.exists(model_dir):
        return None, None, f"Model directory not found: {model_dir}"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()
        return tokenizer, model, None
    except Exception as exc:
        return None, None, f"Failed to load model: {exc}"


def split_into_chunks(text: str, tokenizer, max_len: int = MAX_LEN) -> List[List[int]]:
    token_ids = tokenizer.encode(str(text), add_special_tokens=False, truncation=False)
    chunk_size = max_len - 2
    chunks = []
    for i in range(0, len(token_ids), chunk_size):
        chunk_tokens = token_ids[i:i + chunk_size]
        chunk_tokens = [tokenizer.cls_token_id] + chunk_tokens + [tokenizer.sep_token_id]
        chunks.append(chunk_tokens)
    return chunks if chunks else [[tokenizer.cls_token_id, tokenizer.sep_token_id]]


def run_chunk_inference(text: str, tokenizer, model) -> Dict:
    chunks = split_into_chunks(text, tokenizer, max_len=MAX_LEN)
    all_probs = []

    for chunk in chunks:
        attention_mask = [1] * len(chunk)
        pad_len = MAX_LEN - len(chunk)
        input_ids = chunk + [tokenizer.pad_token_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len

        inputs = {
            "input_ids": torch.tensor([input_ids]),
            "attention_mask": torch.tensor([attention_mask]),
        }

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            all_probs.append(probs)

    chunk_probs = np.array(all_probs)
    avg_probs = chunk_probs.mean(axis=0)
    fake_prob = float(avg_probs[1])
    pred_label = int(fake_prob >= DEFAULT_THRESHOLD)

    return {
        "pred_label": pred_label,
        "fake_probability": fake_prob,
        "real_probability": float(avg_probs[0]),
        "num_chunks": len(chunks),
        "chunk_fake_probs": chunk_probs[:, 1].tolist(),
    }


def assign_risk(prob: float) -> str:
    if prob >= 0.75:
        return "High"
    if prob >= 0.50:
        return "Medium"
    return "Low"


def recommend_action(risk: str) -> str:
    if risk == "High":
        return "Immediate fact-check"
    if risk == "Medium":
        return "Human review"
    return "No action"


def heuristic_predict(text: str) -> Dict:
    suspicious_terms = [
        "shocking", "must read", "they don't want you to know", "viral",
        "breaking", "exposed", "secret", "miracle", "hoax", "cover-up",
    ]
    text_l = text.lower()
    score = 0.08
    for term in suspicious_terms:
        if term in text_l:
            score += 0.12
    if text.isupper():
        score += 0.10
    if text.count("!") >= 3:
        score += 0.10
    score = min(score, 0.95)
    return {
        "pred_label": int(score >= DEFAULT_THRESHOLD),
        "fake_probability": score,
        "real_probability": 1 - score,
        "num_chunks": max(1, len(text.split()) // 380 + 1),
        "chunk_fake_probs": [score],
    }


def predict_document(text: str, model_dir: str) -> Tuple[Dict, str]:
    tokenizer, model, err = load_model_bundle(model_dir)
    if tokenizer is not None and model is not None:
        return run_chunk_inference(text, tokenizer, model), "model"
    return heuristic_predict(text), f"heuristic ({err})"


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🧠 Fake News Analytics Studio")
st.sidebar.caption("Chunk-based BERT + document-level aggregation")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Single Article Prediction",
        "Batch Prediction",
        "Model Analytics",
        "Data Story",
        "About Deployment",
    ],
)

model_dir_input = st.sidebar.text_input("Model directory", value=MODEL_DIR)
st.sidebar.markdown("---")
st.sidebar.write("**Current threshold**")
threshold = st.sidebar.slider("Fake probability threshold", 0.30, 0.90, DEFAULT_THRESHOLD, 0.05)
DEFAULT_THRESHOLD = threshold


# -----------------------------
# Overview
# -----------------------------
if page == "Overview":
    st.title("Fake News Detection Web App")
    st.caption("A smart Streamlit dashboard for fake news classification, analytics, and decision support.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{DEFAULT_METRICS.accuracy:.2%}")
    c2.metric("Precision", f"{DEFAULT_METRICS.precision:.2%}")
    c3.metric("Recall", f"{DEFAULT_METRICS.recall:.2%}")
    c4.metric("F1 Score", f"{DEFAULT_METRICS.f1:.2%}")

    st.markdown("---")
    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("What this app does")
        st.markdown(
            """
            - Classifies news articles as **Real** or **Fake**
            - Supports **full-document prediction** using chunking for long articles
            - Shows **risk level** and **recommended action**
            - Includes **model analytics**, training history, and confidence visualizations
            - Supports **single prediction** and **CSV batch inference**
            """
        )

    with right:
        sample_df = get_sample_predictions()
        st.subheader("Example outputs")
        st.dataframe(sample_df, use_container_width=True, hide_index=True)


# -----------------------------
# Single Prediction
# -----------------------------
elif page == "Single Article Prediction":
    st.title("Single Article Prediction")
    st.caption("Paste a news article or headline + article body. The app predicts fake-news probability and suggested action.")

    user_text = st.text_area(
        "Article content",
        height=280,
        placeholder="Paste article title and body here...",
    )

    col_a, col_b = st.columns([1, 5])
    predict_clicked = col_a.button("Predict", type="primary")
    clear_clicked = col_b.button("Load sample article")

    if clear_clicked and not user_text:
        st.info("Paste sample text manually or click Predict after entering content.")

    if predict_clicked:
        if not user_text.strip():
            st.warning("Please paste article content first.")
        else:
            with st.spinner("Running inference..."):
                result, mode = predict_document(user_text, model_dir_input)

            risk = assign_risk(result["fake_probability"])
            action = recommend_action(risk)
            label_text = "Fake" if result["pred_label"] == 1 else "Real"

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Prediction", label_text)
            m2.metric("Fake probability", f"{result['fake_probability']:.2%}")
            m3.metric("Risk level", risk)
            m4.metric("Chunks used", result["num_chunks"])

            st.success(f"Inference mode: **{mode}**")

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["fake_probability"] * 100,
                title={"text": "Fake News Probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "steps": [
                        {"range": [0, 50], "color": "rgba(34,197,94,0.35)"},
                        {"range": [50, 75], "color": "rgba(250,204,21,0.35)"},
                        {"range": [75, 100], "color": "rgba(239,68,68,0.35)"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.8, "value": DEFAULT_THRESHOLD * 100},
                }
            ))
            gauge.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(gauge, use_container_width=True)

            c1, c2 = st.columns([1.2, 1])
            with c1:
                st.subheader("Recommended action")
                st.info(action)
                st.markdown(
                    f"**Why this matters:** with a threshold of **{DEFAULT_THRESHOLD:.2f}**, this article is classified as **{label_text}** and assigned **{risk} risk**."
                )

            with c2:
                st.subheader("Chunk-level fake probabilities")
                chunk_df = pd.DataFrame({
                    "chunk": list(range(1, len(result["chunk_fake_probs"]) + 1)),
                    "fake_probability": result["chunk_fake_probs"],
                })
                fig = px.bar(chunk_df, x="chunk", y="fake_probability", range_y=[0, 1])
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Batch Prediction
# -----------------------------
elif page == "Batch Prediction":
    st.title("Batch Prediction")
    st.caption("Upload a CSV with a text column and generate predictions in bulk.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    text_col = st.text_input("Name of text column", value="content")

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        st.write("Preview")
        st.dataframe(batch_df.head(), use_container_width=True)

        if st.button("Run batch prediction", type="primary"):
            if text_col not in batch_df.columns:
                st.error(f"Column '{text_col}' not found in uploaded CSV.")
            else:
                outputs = []
                progress = st.progress(0)

                for idx, text in enumerate(batch_df[text_col].fillna("")):
                    pred, _ = predict_document(str(text), model_dir_input)
                    risk = assign_risk(pred["fake_probability"])
                    action = recommend_action(risk)
                    outputs.append(
                        {
                            "predicted_label": "Fake" if pred["pred_label"] == 1 else "Real",
                            "fake_probability": pred["fake_probability"],
                            "risk_category": risk,
                            "recommended_action": action,
                            "num_chunks": pred["num_chunks"],
                        }
                    )
                    progress.progress((idx + 1) / len(batch_df))

                result_df = pd.concat([batch_df.reset_index(drop=True), pd.DataFrame(outputs)], axis=1)
                st.success("Batch prediction completed.")
                st.dataframe(result_df.head(20), use_container_width=True)
                st.download_button(
                    "Download results CSV",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="fake_news_predictions.csv",
                    mime="text/csv",
                )


# -----------------------------
# Model Analytics
# -----------------------------
elif page == "Model Analytics":
    st.title("Model Analytics")
    history = get_training_history()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(history, x="epoch", y=["training_loss", "validation_loss"], markers=True)
        fig.update_layout(title="Training vs Validation Loss", yaxis_title="Loss")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.line(history, x="epoch", y=["accuracy", "f1", "precision", "recall"], markers=True)
        fig.update_layout(title="Validation Metrics by Epoch", yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Confusion Matrix")
    cm_df = get_confusion_matrix_df()
    fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Performance Summary")
    perf_df = pd.DataFrame(
        {
            "metric": ["Accuracy", "Precision", "Recall", "F1 Score", "Eval Loss"],
            "value": [
                DEFAULT_METRICS.accuracy,
                DEFAULT_METRICS.precision,
                DEFAULT_METRICS.recall,
                DEFAULT_METRICS.f1,
                DEFAULT_METRICS.eval_loss,
            ],
        }
    )
    fig = px.bar(perf_df, x="metric", y="value", text="value", range_y=[0, 1.05])
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Data Story
# -----------------------------
elif page == "Data Story":
    st.title("Data Story & Analytics")

    st.subheader("1. Descriptive Analytics — What happened?")
    st.write(
        "The WELFake dataset contains more than 72,000 news articles and is nearly balanced between real and fake news. "
        "Article lengths vary widely, with many long articles exceeding BERT's 512-token limit."
    )

    st.subheader("2. Diagnostic Analytics — Why did it happen?")
    st.write(
        "The main challenge was that standard BERT can only process 512 tokens per input. "
        "To avoid truncating long articles, the app uses chunking and then aggregates chunk-level predictions into one final document decision."
    )

    st.subheader("3. Predictive Analytics — What is likely to happen?")
    st.write(
        "The fine-tuned BERT model predicts whether a news article is real or fake. "
        f"In prior experiments, the model achieved about {DEFAULT_METRICS.f1:.2%} F1-score on unseen test data."
    )

    st.subheader("4. Prescriptive Analytics — What should be done?")
    st.write(
        "The app transforms probabilities into actions. Low-risk articles require no action, medium-risk articles are routed for human review, and high-risk articles are prioritized for immediate fact-checking."
    )

    st.subheader("Pipeline Overview")
    st.markdown(
        """
        1. Load article text  
        2. Tokenize with BERT tokenizer  
        3. Split long documents into chunks  
        4. Score each chunk with BERT  
        5. Average chunk probabilities  
        6. Assign final label, risk, and action
        """
    )


# -----------------------------
# Deployment Notes
# -----------------------------
elif page == "About Deployment":
    st.title("GitHub + Streamlit Deployment Guide")
    st.markdown(
        """
        ### Recommended repo structure
        ```
        fake-news-app/
        ├── app.py
        ├── requirements.txt
        ├── README.md
        ├── bert_fake_news/   # fine-tuned model directory
        └── assets/
        ```

        ### Example requirements.txt
        ```
        streamlit
        pandas
        numpy
        plotly
        scikit-learn
        transformers
        torch
        seaborn
        ```

        ### Deployment flow
        1. Push code to GitHub  
        2. Connect repo to Streamlit Community Cloud  
        3. Set `app.py` as entry point  
        4. Add model files or configure `MODEL_DIR`  
        5. Deploy
        ```
        """
    )

    st.info(
        "Tip: if model files are too large for GitHub, store them externally and download them at startup, or use Git LFS / Hugging Face Hub."
    )
