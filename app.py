import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title="Fake News Detection AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_NAME = "laitali2026/fake-news-bert-2epoch-model"
MAX_LEN = 512
DEFAULT_THRESHOLD = 0.50

MODEL_METRICS = {
    "Accuracy": 0.9890,
    "Precision": 0.9882,
    "Recall": 0.9901,
    "F1 Score": 0.9892,
}

CONFUSION_MATRIX = pd.DataFrame(
    [[974, 12], [10, 1004]],
    index=["Actual Real", "Actual Fake"],
    columns=["Predicted Real", "Predicted Fake"],
)

TRAINING_HISTORY = pd.DataFrame(
    {
        "Epoch": [1, 2],
        "Training Loss": [0.1576, 0.0652],
        "Validation Loss": [0.1689, 0.2098],
        "Accuracy": [0.9578, 0.9599],
        "Precision": [0.9438, 0.9719],
        "Recall": [0.9700, 0.9439],
        "F1": [0.9567, 0.9577],
    }
)

st.markdown(
    """
    <style>
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
    .good {color: #16a34a; font-weight: 800;}
    .bad {color: #dc2626; font-weight: 800;}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def chunk_document(text, tokenizer, max_len=512):
    token_ids = tokenizer.encode(str(text), add_special_tokens=False, truncation=False)
    chunk_size = max_len - 2
    chunks = []

    for i in range(0, len(token_ids), chunk_size):
        chunk_tokens = token_ids[i:i + chunk_size]
        chunk_tokens = [tokenizer.cls_token_id] + chunk_tokens + [tokenizer.sep_token_id]
        chunks.append(chunk_tokens)

    if not chunks:
        chunks = [[tokenizer.cls_token_id, tokenizer.sep_token_id]]

    return chunks


def assign_risk(prob):
    if prob >= 0.75:
        return "High Risk"
    if prob >= 0.50:
        return "Medium Risk"
    return "Low Risk"


def recommend_action(risk):
    if risk == "High Risk":
        return "Immediate fact-checking recommended"
    if risk == "Medium Risk":
        return "Send to human reviewer"
    return "No immediate action needed"


def predict_article(text, threshold):
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

    # Use max probability so one suspicious chunk can flag the document.
    final_fake_prob = max_fake_prob

    predicted_label = int(final_fake_prob >= threshold)

    risk = assign_risk(final_fake_prob)
    action = recommend_action(risk)

    return {
        "prediction": "Fake News" if predicted_label == 1 else "Real News",
        "predicted_label": predicted_label,
        "fake_probability": final_fake_prob,
        "avg_fake_probability": avg_fake_prob,
        "max_fake_probability": max_fake_prob,
        "real_probability": 1 - final_fake_prob,
        "num_chunks": len(chunks),
        "chunk_fake_probs": fake_probs,
        "risk": risk,
        "action": action,
    }


def probability_gauge(fake_prob, threshold):
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


def chunk_chart(chunk_probs):
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

if page == "Home":
    st.markdown('<div class="main-title">Fake News Detection AI Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Fine-tuned 2-epoch BERT model with chunk-based document processing.</div>',
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
            "This app detects fake news using a fine-tuned BERT transformer model. "
            "Long articles are split into 512-token chunks, each chunk is classified, "
            "and the chunk probabilities are combined into a final document-level prediction."
        )

    with right:
        st.subheader("Pipeline")
        st.markdown(
            """
            1. Enter or upload article text  
            2. Tokenize using BERT tokenizer  
            3. Split long text into 512-token chunks  
            4. Predict each chunk  
            5. Use the most suspicious chunk probability  
            6. Assign risk level and action
            """
        )

elif page == "Single Article Prediction":
    st.title("Single Article Prediction")
    st.caption("Paste a full news article or headline + body text.")

    article_text = st.text_area("Article Text", height=300)

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

            st.subheader("Debug: Model Probability Details")
            st.write("Model used:", MODEL_NAME)
            st.write("Average fake probability:", result["avg_fake_probability"])
            st.write("Maximum fake probability:", result["max_fake_probability"])
            st.write("Chunk fake probabilities:", result["chunk_fake_probs"])

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
                        "avg_fake_probability": pred["avg_fake_probability"],
                        "max_fake_probability": pred["max_fake_probability"],
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

elif page == "Model Dashboard":
    st.title("Model Dashboard — 2 Epoch BERT Model")

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

elif page == "Four Analytics":
    st.title("Four Analytics Framework")

    st.header("1. Descriptive Analytics — What happened?")
    st.write(
        """
        The WELFake dataset contains more than 72,000 labeled articles.
        After cleaning missing values, approximately 71,500 articles remained.
        The title and article body were combined into a single content field.
        """
    )

    st.header("2. Diagnostic Analytics — Why did it happen?")
    st.write(
        """
        Many articles exceeded BERT's 512-token input limit.
        To solve this, articles were split into 512-token chunks so the model could process the full document.
        """
    )

    st.header("3. Predictive Analytics — What is likely to happen?")
    st.write(
        """
        The fine-tuned BERT model predicts the probability that an article is fake.
        Each chunk is scored, and the most suspicious chunk is used to support the final risk decision.
        """
    )

    st.header("4. Prescriptive Analytics — What should be done?")
    st.write(
        """
        Predictions are converted into risk levels:
        Low Risk means no action, Medium Risk means human review, and High Risk means immediate fact-checking.
        """
    )

elif page == "About":
    st.title("About This Project")
    st.write(
        "This project uses a fine-tuned BERT model for fake news detection. "
        "The model is hosted on Hugging Face and the web app is deployed with Streamlit."
    )

    st.subheader("Model")
    st.code(MODEL_NAME)
