import os
import re
import json

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report

# ========================================================
# STREAMLIT SETUP
# ========================================================
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

st.title("📘 Sentiment Analysis Explorer")
st.write("EDA • Model Information • Test Streaming • XAI • Model Comparison")

# Ensure session state key exists
if "stream_idx" not in st.session_state:
    st.session_state["stream_idx"] = 0

# ========================================================
# CONSTANTS
# ========================================================
MODEL_NAMES = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "roberta-base",
    "xlnet-base-cased",
]

LABEL_MAP = {0: "negative", 1: "positive"}

PREDICTION_FILES = {
    "distilbert-base-uncased": "distilbert-base-uncased_predictions.csv",
    "bert-base-uncased": "bert-base-uncased_predictions.csv",
    "roberta-base": "roberta-base_predictions.csv",
    "xlnet-base-cased": "xlnet-base-cased_predictions.csv",
}

# ========================================================
# UTILS
# ========================================================
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_sentiment_col(df: pd.DataFrame):
    """
    Try to detect the sentiment/label column in a robust way.
    """
    candidates = ["sentiment", "label", "target", "polarity"]
    for c in df.columns:
        if c.lower() in candidates:
            return c
    return None


@st.cache_data(show_spinner=False)
def load_csv_from_upload(uploaded_file):
    # Cache based on file content; Streamlit handles this safely
    return pd.read_csv(uploaded_file)


@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer(model_choice: str):
    """
    Load a finetuned model and tokenizer from local folder.
    Cached so it only loads once per model.
    """
    folder = f"./{model_choice}-finetuned"
    tokenizer = AutoTokenizer.from_pretrained(folder)
    model = AutoModelForSequenceClassification.from_pretrained(folder)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device


def compute_char_importance_from_token_scores(text: str, offsets, token_scores):
    """
    Map token-level scores to character-level importance over the original text.
    """
    char_importance = np.zeros(len(text), dtype=float)
    for (start, end), score in zip(offsets, token_scores):
        if start == end:
            continue
        # Use max to keep strongest contribution for overlapping tokens
        char_importance[start:end] = np.maximum(char_importance[start:end], score)
    return char_importance


def char_to_word_importance(text: str, char_importance):
    """
    Aggregate character-level importance to word-level importance.
    """
    words = text.split()
    word_scores = []
    idx = 0  # char index

    for w in words:
        w_len = len(w)
        # Average importance over the word span
        span_imp = np.mean(char_importance[idx:idx + w_len]) if w_len > 0 else 0.0
        word_scores.append((w, float(span_imp)))
        idx += w_len + 1  # +1 for the space

    return word_scores


def get_grad_word_importance(model, tokenizer, text: str, device: str):
    """
    Gradient × Embedding word-level importance on ORIGINAL text.
    Returns (word_scores, char_importance).
    """
    model.eval()

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256,
        return_offsets_mapping=True,
    )

    offsets = encoded["offset_mapping"][0].tolist()
    encoded = {k: v for k, v in encoded.items() if k != "offset_mapping"}

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # 1. Extract embeddings
    embedding_layer = model.get_input_embeddings()
    embeddings = embedding_layer(input_ids)
    embeddings.retain_grad()

    # 2. Forward pass using embeddings
    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    logits = outputs.logits
    pred_class = torch.argmax(logits, dim=1)

    # 3. Backprop on predicted logit
    model.zero_grad()
    logits[0, pred_class.item()].backward()

    # 4. Token-level importance = gradient norm
    grads = embeddings.grad[0]  # (seq_len, hidden_dim)
    scores = grads.norm(dim=1).detach().cpu().numpy()

    # 5. Normalize scores
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    # 6. Map to chars -> words
    char_importance = compute_char_importance_from_token_scores(text, offsets, scores)
    word_scores = char_to_word_importance(text, char_importance)

    return word_scores, char_importance


def get_lrp_word_importance(model, tokenizer, text: str, device: str):
    """
    Approximate LRP-style word importance using gradient * activation
    on the last hidden layer.
    Returns (word_scores, char_importance).
    """
    model.eval()

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256,
        return_offsets_mapping=True,
    )

    offsets = encoded["offset_mapping"][0].tolist()
    encoded = {k: v for k, v in encoded.items() if k != "offset_mapping"}

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )

    logits = outputs.logits
    pred_class = torch.argmax(logits, dim=1)

    last_hidden = outputs.hidden_states[-1]  # (batch, seq, hidden)
    last_hidden.retain_grad()

    model.zero_grad()
    logits[0, pred_class.item()].backward()

    hidden_grads = last_hidden.grad[0]  # (seq, hidden)
    hidden_vals = last_hidden[0]        # (seq, hidden)

    # Relevance ~ |activation * gradient|
    token_scores = (hidden_vals * hidden_grads).sum(dim=1).abs().detach().cpu().numpy()
    token_scores = (token_scores - token_scores.min()) / (token_scores.max() - token_scores.min() + 1e-8)

    char_importance = compute_char_importance_from_token_scores(text, offsets, token_scores)
    word_scores = char_to_word_importance(text, char_importance)

    return word_scores, char_importance


def get_attention_word_importance(model, tokenizer, text: str, device: str):
    """
    Attention-based word importance.
    Uses CLS (or first token) attention averaged across layers and heads.
    Returns (word_scores, char_importance).
    """
    model.eval()

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256,
        return_offsets_mapping=True,
    )

    offsets = encoded["offset_mapping"][0].tolist()
    encoded = {k: v for k, v in encoded.items() if k != "offset_mapping"}

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
    )

    if outputs.attentions is None:
        raise RuntimeError("Model did not return attentions. Check config or model type.")

    # attentions: tuple(layers) of (batch, heads, seq, seq)
    att_stack = torch.stack(outputs.attentions, dim=0)  # (layers, batch, heads, seq, seq)
    # Take attention from first token (CLS / <s>) to all tokens
    cls_att = att_stack[:, 0, :, 0, :]  # (layers, heads, seq)
    token_scores = cls_att.mean(dim=(0, 1))  # (seq,)
    token_scores = token_scores.detach().cpu().numpy()

    # Normalize
    token_scores = (token_scores - token_scores.min()) / (token_scores.max() - token_scores.min() + 1e-8)

    char_importance = compute_char_importance_from_token_scores(text, offsets, token_scores)
    word_scores = char_to_word_importance(text, char_importance)

    return word_scores, char_importance


def get_sentence_importance(text: str, char_importance):
    """
    Split text into sentences and aggregate character importance per sentence.
    Returns list of (sentence, importance).
    """
    sentences = []
    # Simple regex-based sentence segmentation
    for m in re.finditer(r"[^.!?]+[.!?]?", text):
        sent = m.group(0).strip()
        if not sent:
            continue
        start, end = m.start(), m.end()
        if end > start:
            score = float(char_importance[start:end].mean())
        else:
            score = 0.0
        sentences.append((sent, score))

    return sentences


def render_word_importance_html(word_scores, cmap="red"):
    """
    Render HTML span highlights for words based on importance.
    """
    html = ""
    for word, score in word_scores:
        # Clamp score in [0,1]
        score = max(0.0, min(1.0, float(score)))
        if cmap == "red":
            color = f"rgba(255, 0, 0, {score})"
        elif cmap == "blue":
            color = f"rgba(0, 0, 255, {score})"
        elif cmap == "green":
            color = f"rgba(0, 200, 0, {score})"
        else:
            color = f"rgba(255, 0, 0, {score})"
        html += f"<span style='background-color:{color}'>{word}</span> "
    return html


def plot_word_heatmap(word_scores, title: str):
    """
    Plot a simple heatmap of word importance.
    """
    if not word_scores:
        st.info("No words to visualize.")
        return

    words, scores = zip(*word_scores)
    scores = np.array(scores, dtype=float)

    fig, ax = plt.subplots(figsize=(min(12, max(6, len(words) * 0.4)), 2.5))
    im = ax.imshow(scores[np.newaxis, :], aspect="auto")

    ax.set_yticks([])
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha="right")
    ax.set_title(title)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig, clear_figure=True)


def plot_sentiment_distribution(series: pd.Series):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=series, ax=ax)
    ax.set_title("Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    st.pyplot(fig, clear_figure=True)


def plot_confusion_matrix(y_true, y_pred, labels=("negative", "positive")):
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig, clear_figure=True)


# ========================================================
# SECTION 1: EDA
# ========================================================
def section_eda():
    st.header("📊 Exploratory Data Analysis")

    train_file = st.file_uploader("Upload train.csv", type=["csv"], key="train_eda")
    test_file = st.file_uploader("Upload test.csv", type=["csv"], key="test_eda")

    # Use tabs for train / test EDA
    tab_train, tab_test = st.tabs(["Train EDA", "Test EDA"])

    # ------------------------
    # TRAIN EDA
    # ------------------------
    with tab_train:
        if train_file:
            st.subheader("Train Data Overview")

            try:
                train_df = load_csv_from_upload(train_file)
            except Exception as e:
                st.error("❌ Error reading train.csv")
                st.write(e)
                return

            st.write("First 5 rows:")
            st.write(train_df.head())
            st.write("Columns:", list(train_df.columns))

            # Ensure Review column exists
            if "Review" not in train_df.columns:
                st.error("❌ Column 'Review' not found in train.csv")
                return

            # Clean reviews
            with st.spinner("Cleaning text..."):
                train_df["clean_review"] = train_df["Review"].astype(str).apply(clean_text)

            st.subheader("Missing Values")
            st.write(train_df.isnull().sum())

            st.subheader("Sentiment Distribution")

            sent_col = find_sentiment_col(train_df)
            if not sent_col:
                st.error(
                    "❌ Could not detect a sentiment/label column.\n\n"
                    "Expected one of: Sentiment, Label, Target, Polarity"
                )
                return

            series = train_df[sent_col].astype(str).str.strip()
            series = series[series != ""]

            if series.empty:
                st.warning("⚠ Sentiment column is empty after cleaning.")
            else:
                st.write("Value counts:")
                st.write(series.value_counts())
                plot_sentiment_distribution(series)
        else:
            st.info("📥 Upload train.csv to see EDA.")

    # ------------------------
    # TEST EDA
    # ------------------------
    with tab_test:
        if test_file:
            st.subheader("Test Data Overview")

            try:
                test_df = load_csv_from_upload(test_file)
            except Exception as e:
                st.error("❌ Error reading test.csv")
                st.write(e)
                return

            st.write("First 5 rows:")
            st.write(test_df.head())
            st.write("Columns:", list(test_df.columns))

            st.subheader("Missing Values (Test)")
            st.write(test_df.isnull().sum())
        else:
            st.info("📥 Upload test.csv to see EDA for test data.")


# ========================================================
# SECTION 2: MODEL INFORMATION
# ========================================================
def section_model_information():
    st.header("🧠 Model Information")

    model_choice = st.selectbox("Select Model", MODEL_NAMES)
    model_folder = f"./{model_choice}-finetuned"
    trainer_state_path = os.path.join(model_folder, "trainer_state.json")
    config_path = os.path.join(model_folder, "config.json")
    pred_file_path = PREDICTION_FILES.get(model_choice, "")

    st.write("Model folder:", model_folder)
    st.write("Prediction CSV:", pred_file_path)

    if not os.path.exists(model_folder):
        st.error(f"❌ Model folder not found: {model_folder}")
        return

    try:
        files = os.listdir(model_folder)
    except Exception as e:
        st.error("❌ Unable to list files in model folder.")
        st.write(e)
        return

    st.success("✅ Model folder found.")
    with st.expander("Show files in model folder"):
        st.write(files)

    # ------------------------
    # CONFIG
    # ------------------------
    if os.path.exists(config_path):
        st.subheader("Model Config")
        try:
            with open(config_path) as f:
                st.json(json.load(f))
        except Exception as e:
            st.error("❌ Error reading config.json")
            st.write(e)

    # ------------------------
    # TRAIN LOGS
    # ------------------------
    if os.path.exists(trainer_state_path):
        st.subheader("Training Logs")

        try:
            with open(trainer_state_path) as f:
                logs = json.load(f).get("log_history", [])
            df_logs = pd.DataFrame(logs)
        except Exception as e:
            st.error("❌ Error reading trainer_state.json")
            st.write(e)
            df_logs = None

        if df_logs is not None and not df_logs.empty:
            st.dataframe(df_logs.head())

            # Plot metrics if available
            if "epoch" in df_logs.columns:
                col1, col2 = st.columns(2)

                with col1:
                    if "eval_accuracy" in df_logs.columns:
                        fig, ax = plt.subplots()
                        ax.plot(df_logs["epoch"], df_logs["eval_accuracy"], marker="o")
                        ax.set_xlabel("Epoch")
                        ax.set_ylabel("Eval Accuracy")
                        ax.set_title("Eval Accuracy over Epochs")
                        st.pyplot(fig, clear_figure=True)

                with col2:
                    if "eval_loss" in df_logs.columns:
                        fig, ax = plt.subplots()
                        ax.plot(df_logs["epoch"], df_logs["eval_loss"], marker="o")
                        ax.set_xlabel("Epoch")
                        ax.set_ylabel("Eval Loss")
                        ax.set_title("Eval Loss over Epochs")
                        st.pyplot(fig, clear_figure=True)
        else:
            st.info("ℹ No training logs found in trainer_state.json")
    else:
        st.info("ℹ trainer_state.json not found; skipping training log display.")

    # ------------------------
    # TEST METRICS
    # ------------------------
    st.subheader("Test Performance")

    if pred_file_path and os.path.exists(pred_file_path):
        try:
            df_pred = pd.read_csv(pred_file_path)
        except Exception as e:
            st.error("❌ Error reading prediction CSV.")
            st.write(e)
            return

        if not {"Sentiment", "Prediction"}.issubset(df_pred.columns):
            st.error("❌ Prediction CSV must contain 'Sentiment' and 'Prediction' columns.")
            st.write("Found columns:", list(df_pred.columns))
            return

        y_true = df_pred["Sentiment"]
        y_pred = df_pred["Prediction"]

        acc = (y_true == y_pred).mean()
        st.success(f"✅ Test Accuracy: {acc:.4f}")

        st.write("Classification Report")
        report_df = pd.DataFrame(
            classification_report(y_true, y_pred, output_dict=True)
        ).transpose()
        st.dataframe(report_df)

        st.write("Confusion Matrix")
        plot_confusion_matrix(y_true, y_pred, labels=("negative", "positive"))
    else:
        st.warning(f"⚠ Prediction CSV not found: {pred_file_path}")


# ========================================================
# SECTION 3: TEST STREAMING + XAI
# ========================================================
def section_test_streaming():
    st.header("⏩ Test Streaming with Explainability")

    test_file = st.file_uploader("Upload test.csv", type=["csv"], key="test_stream")

    if not test_file:
        st.info("📥 Upload test.csv to start streaming.")
        return

    try:
        test_df = load_csv_from_upload(test_file)
    except Exception as e:
        st.error("❌ Error reading test.csv")
        st.write(e)
        return

    if "Review" not in test_df.columns:
        st.error("❌ Column 'Review' not found in test.csv")
        st.write("Columns:", list(test_df.columns))
        return

    test_df["clean_review"] = test_df["Review"].astype(str).apply(clean_text)

    model_choice = st.selectbox("Model for Streaming", MODEL_NAMES)
    model, tokenizer, device = load_model_and_tokenizer(model_choice)

    idx = st.session_state["stream_idx"]

    if idx >= len(test_df):
        st.success("✅ Done streaming all examples!")
        if st.button("Restart"):
            st.session_state["stream_idx"] = 0
            st.rerun()
        return

    row = test_df.iloc[idx]
    text = row["Review"]
    clean = row["clean_review"]

    st.subheader(f"Review {idx + 1}/{len(test_df)}")
    st.write(text)

    # Prediction
    with torch.no_grad():
        encoded = tokenizer(
            clean,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256,
        )

        logits = model(
            encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
        ).logits
        pred = int(torch.argmax(logits))
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

    st.markdown(f"### Prediction: **{LABEL_MAP.get(pred, pred)}**")
    st.write(f"Class probabilities: {probs}")

    if "Sentiment" in test_df.columns:
        st.markdown(f"### Actual: **{row['Sentiment']}**")

    # XAI
    st.subheader("🧠 XAI Explanations")

    with st.spinner("Computing explanations (Grad × Emb, LRP, Attention, Sentences)..."):
        # Gradient × Embedding
        grad_word_scores, grad_char_imp = get_grad_word_importance(model, tokenizer, clean, device)

        # Approximate LRP
        try:
            lrp_word_scores, lrp_char_imp = get_lrp_word_importance(model, tokenizer, clean, device)
        except Exception as e:
            lrp_word_scores, lrp_char_imp = [], None
            st.warning("⚠ LRP-style explanation failed:")
            st.write(e)

        # Attention
        try:
            attn_word_scores, attn_char_imp = get_attention_word_importance(model, tokenizer, clean, device)
        except Exception as e:
            attn_word_scores, attn_char_imp = [], None
            st.warning("⚠ Attention-based explanation failed:")
            st.write(e)

        # Sentence-level (from gradient char importance)
        sent_scores = get_sentence_importance(clean, grad_char_imp)

    # Tabs for different XAI views
    tab_words, tab_heat, tab_sent, tab_lrp, tab_attn = st.tabs(
        [
            "Word Highlights (Grad × Emb)",
            "Heatmap (Grad × Emb)",
            "Sentence-level",
            "LRP-style (approx)",
            "Attention-based",
        ]
    )

    # ------------------------
    # TAB 1: WORD HIGHLIGHTS
    # ------------------------
    with tab_words:
        st.markdown("#### Word-level importance (Gradient × Embedding)")
        df_xai = pd.DataFrame(grad_word_scores, columns=["Word", "Importance"])
        with st.expander("Show word importance table"):
            st.dataframe(df_xai)

        html = render_word_importance_html(grad_word_scores, cmap="red")
        st.markdown(html, unsafe_allow_html=True)

    # ------------------------
    # TAB 2: HEATMAP
    # ------------------------
    with tab_heat:
        st.markdown("#### Heatmap of word importance (Gradient × Embedding)")
        plot_word_heatmap(grad_word_scores, "Grad × Embedding Word Importance")

    # ------------------------
    # TAB 3: SENTENCE-LEVEL
    # ------------------------
    with tab_sent:
        st.markdown("#### Sentence-level explanation (from Grad × Embedding)")

        if sent_scores:
            df_sent = pd.DataFrame(sent_scores, columns=["Sentence", "Importance"])
            st.dataframe(df_sent)

            # Highlight most important sentence(s)
            top_sentences = sorted(sent_scores, key=lambda x: x[1], reverse=True)[:3]
            st.markdown("**Top sentences by importance:**")
            for s, sc in top_sentences:
                st.markdown(f"- ({sc:.3f}) {s}")
        else:
            st.info("No sentences detected in this review.")

    # ------------------------
    # TAB 4: LRP-STYLE
    # ------------------------
    with tab_lrp:
        st.markdown("#### Approximate LRP-style word importance")
        if lrp_word_scores:
            df_lrp = pd.DataFrame(lrp_word_scores, columns=["Word", "Importance"])
            with st.expander("Show LRP-style word importance table"):
                st.dataframe(df_lrp)

            html_lrp = render_word_importance_html(lrp_word_scores, cmap="blue")
            st.markdown(html_lrp, unsafe_allow_html=True)

            plot_word_heatmap(lrp_word_scores, "LRP-style Word Importance")
        else:
            st.info("LRP-style explanation not available for this example.")

    # ------------------------
    # TAB 5: ATTENTION-BASED
    # ------------------------
    with tab_attn:
        st.markdown("#### Attention-based word importance (CLS → Tokens)")

        if attn_word_scores:
            df_attn = pd.DataFrame(attn_word_scores, columns=["Word", "Importance"])
            with st.expander("Show attention-based word importance table"):
                st.dataframe(df_attn)

            html_attn = render_word_importance_html(attn_word_scores, cmap="green")
            st.markdown(html_attn, unsafe_allow_html=True)

            plot_word_heatmap(attn_word_scores, "Attention-based Word Importance")
        else:
            st.info("Attention-based explanation not available for this example.")

    # Navigation buttons
    col_prev, col_next = st.columns(2)
    with col_next:
        if st.button("Next ➜"):
            st.session_state["stream_idx"] += 1
            st.rerun()
    with col_prev:
        if st.button("⟵ Previous", disabled=(idx == 0)):
            st.session_state["stream_idx"] = max(0, idx - 1)
            st.rerun()


# ========================================================
# SECTION 4: MODEL COMPARISON
# ========================================================
def section_model_comparison():
    st.header("📊 Model Comparison")

    rows = []

    for m in MODEL_NAMES:
        pred_file = PREDICTION_FILES.get(m, "")
        if not pred_file or not os.path.exists(pred_file):
            st.warning(f"⚠ Skipping {m}: prediction file not found ({pred_file})")
            continue

        df = pd.read_csv(pred_file)
        if not {"Sentiment", "Prediction"}.issubset(df.columns):
            st.warning(f"⚠ Skipping {m}: missing 'Sentiment' or 'Prediction' columns.")
            continue

        y_true = df["Sentiment"]
        y_pred = df["Prediction"]

        report = classification_report(y_true, y_pred, output_dict=True)

        rows.append(
            {
                "Model": m,
                "Accuracy": (y_true == y_pred).mean(),
                "Precision_Pos": report["positive"]["precision"],
                "Recall_Pos": report["positive"]["recall"],
                "F1_Pos": report["positive"]["f1-score"],
                "Precision_Neg": report["negative"]["precision"],
                "Recall_Neg": report["negative"]["recall"],
                "F1_Neg": report["negative"]["f1-score"],
            }
        )

    if not rows:
        st.error("❌ No valid prediction CSVs found for any model.")
        return

    df_comp = pd.DataFrame(rows)
    st.dataframe(df_comp)

    fig, ax = plt.subplots()
    sns.barplot(data=df_comp, x="Model", y="Accuracy", ax=ax)
    ax.set_title("Model Accuracy Comparison")
    st.pyplot(fig, clear_figure=True)

    if st.button("Export Comparison CSV"):
        out_path = "model_comparison_metrics.csv"
        df_comp.to_csv(out_path, index=False)
        st.success(f"✅ Saved comparison metrics to {out_path}")


# ========================================================
# SIDEBAR NAVIGATION
# ========================================================
st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Go To:",
    ["EDA", "Model Information", "Test Streaming", "Model Comparison"],
)

# ========================================================
# MAIN ROUTING
# ========================================================
if section == "EDA":
    section_eda()
elif section == "Model Information":
    section_model_information()
elif section == "Test Streaming":
    section_test_streaming()
elif section == "Model Comparison":
    section_model_comparison()

