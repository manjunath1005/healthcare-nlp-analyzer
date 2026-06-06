
import streamlit as st
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import pickle
import zipfile
import json
import io
import os
from collections import defaultdict
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Cross-version compatibility helpers ───────────────────────────────────────
#
# The .keras files were saved on Python 3.11+ with a newer Keras build.
# Two issues arise when loading on Python 3.10 / older Keras:
#
#  A) quantization_config: newer Keras adds this key to every layer's saved
#     config, but the current __init__ methods don't accept it.  Fix: strip it
#     from the JSON before Keras ever parses the config.
#
#  B) Lambda bytecode: Keras serialises Lambda functions as Python marshal
#     bytecode, which is version-specific.  The saved bytecode is in 3.11+
#     format; Python 3.10 can't load it.  Fix: replace the Lambda layer entry
#     in the JSON with an equivalent custom layer.

def _strip_quantization(obj):
    """Recursively remove quantization_config from a Keras config structure."""
    if isinstance(obj, dict):
        return {k: _strip_quantization(v)
                for k, v in obj.items()
                if k != 'quantization_config'}
    if isinstance(obj, list):
        return [_strip_quantization(v) for v in obj]
    return obj


# Replacement for the Lambda(sparse→dense) layer stored in the model.
_SPARSE_TO_DENSE_CONFIG = {
    "module": None,
    "class_name": "_SparseToDense",
    "config": {},
    "registered_name": "_SparseToDense",
}

def _replace_lambda_layers(obj):
    """Replace Lambda layers in a Keras config with our _SparseToDense stub."""
    if isinstance(obj, dict):
        if obj.get('class_name') == 'Lambda':
            # Preserve the layer name so the graph topology is unchanged.
            name = obj.get('config', {}).get('name', 'sparse_to_dense')
            stub = dict(_SPARSE_TO_DENSE_CONFIG)
            stub['config'] = {'name': name}
            stub['name'] = name
            # Copy over inbound_nodes / build_config so connectivity is intact.
            for key in ('inbound_nodes', 'build_config', 'name'):
                if key in obj:
                    stub[key] = obj[key]
            return stub
        return {k: _replace_lambda_layers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_lambda_layers(v) for v in obj]
    return obj


def _patch_keras_zip(src_path):
    """
    Return a BytesIO of a .keras archive whose config.json has been sanitised:
      • quantization_config stripped from every layer
      • Lambda layers replaced with our _SparseToDense stub
    """
    with open(src_path, 'rb') as fh:
        raw = fh.read()

    buf_in  = io.BytesIO(raw)
    buf_out = io.BytesIO()

    with zipfile.ZipFile(buf_in, 'r') as zin, \
         zipfile.ZipFile(buf_out, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename == 'config.json':
                cfg  = json.loads(data)
                cfg  = _strip_quantization(cfg)
                cfg  = _replace_lambda_layers(cfg)
                data = json.dumps(cfg).encode('utf-8')
            zout.writestr(info, data)

    buf_out.seek(0)
    return buf_out


class _SparseToDense(keras.layers.Layer):
    """Drop-in replacement for Lambda(lambda ts: tf.sparse.to_dense(ts))."""
    def call(self, ts, **kwargs):          # accept mask, training, etc.
        if isinstance(ts, tf.SparseTensor):
            return tf.sparse.to_dense(ts)
        return ts

    def get_config(self):
        return super().get_config()


def load_keras_model_compat(path, custom_objects=None, compile=False):
    """
    Load a .keras model with compatibility fixes applied transparently.
    A patched copy is written to a temp file, loaded, then deleted.
    """
    patched_buf = _patch_keras_zip(path)
    tmp_path = path + '.tmp_compat.keras'
    try:
        with open(tmp_path, 'wb') as fh:
            fh.write(patched_buf.read())
        co = {'_SparseToDense': _SparseToDense}
        if custom_objects:
            co.update(custom_objects)
        model = keras.models.load_model(
            tmp_path,
            custom_objects=co,
            compile=compile,
            safe_mode=False,
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return model
# ──────────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import plotly.express as px
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import tempfile

MAX_LEN = 500
EMBED_DIM = 128

def positional_encoding(max_len, d_model):
    import numpy as np
    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        # Strip kwargs Keras passes during deserialization (trainable, dtype, name…)
        kwargs.pop('trainable', None)
        kwargs.pop('dtype', None)
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        self.pe = tf.constant(positional_encoding(max_len, d_model), dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        pe = self.pe[:seq_len, :]
        pe = tf.expand_dims(pe, axis=0)
        return x + pe

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len, "d_model": self.d_model})
        return config

@st.cache_resource
def load_assets():
    custom_objects = {"PositionalEncoding": PositionalEncoding}

    classifier = load_keras_model_compat(
        "medical_positional_attention.keras",
        custom_objects=custom_objects,
    )

    attention_model = load_keras_model_compat(
        "attention_extractor.keras",
        custom_objects=custom_objects,
    )

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder (1).pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return classifier, attention_model, tokenizer, label_encoder

# st.set_page_config MUST be the first Streamlit command in the script.
st.set_page_config(page_title="Medical Report Understanding System", layout="wide")

st.markdown("""
<style>

.stApp {
    background-color: #f4f8fb;
}

.hero {
    background: linear-gradient(135deg,#2563eb,#0ea5e9);
    padding: 2rem;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}

.term-tag {
    display:inline-block;
    background:#dbeafe;
    color:#1e40af;
    padding:8px 14px;
    border-radius:20px;
    margin:4px;
    font-weight:600;
}

.summary-card {
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0 2px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)
classifier, attention_model, tokenizer, label_encoder = load_assets()

reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}

def explain_prediction(report_text):
    seq = tokenizer.texts_to_sequences([report_text])

    padded = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    probs = classifier.predict(padded, verbose=0)
    pred_idx = int(np.argmax(probs))

    specialty = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(probs[0][pred_idx] * 100)

    scores = attention_model.predict(padded, verbose=0)

    importance = scores.mean(axis=1)[0]
    token_scores = importance.mean(axis=0)

    tokens = padded[0]

    words = []
    for token in tokens:
        if token == 0:
            continue
        words.append(reverse_word_index.get(token, "<OOV>"))

    word_scores = defaultdict(list)

    for word, score in zip(words, token_scores[:len(words)]):
        word_scores[word].append(float(score))

    final_scores = {
        word: np.mean(vals)
        for word, vals in word_scores.items()
    }

    top_words = sorted(
        final_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:15]

    return specialty, confidence, probs[0], top_words

def create_pdf(specialty, confidence, top_words):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(tmp.name)
    styles = getSampleStyleSheet()

    content = [
        Paragraph("Medical Analysis Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Predicted Specialty: {specialty}", styles["BodyText"]),
        Paragraph(f"Confidence: {confidence:.2f}%", styles["BodyText"]),
        Spacer(1, 12),
        Paragraph("Top Influential Medical Terms", styles["Heading2"])
    ]

    for word, score in top_words:
        content.append(
            Paragraph(f"{word} : {score:.4f}", styles["BodyText"])
        )

    doc.build(content)
    return tmp.name



st.markdown("""
<div class="hero">
<h1>🏥 Intelligent Medical Report Understanding System</h1>
<p>Healthcare NLP using Self-Attention + Positional Encoding</p>
</div>
""", unsafe_allow_html=True)
with st.sidebar:

    st.header("📋 User Guide")

    st.write("""
    1. Upload a medical report
    2. Review extracted text
    3. Click Analyze
    4. Download PDF report
    """)

    st.divider()

    st.info(
        "Powered by TensorFlow + Attention Mechanism"
    )

st.subheader("📄 Upload Medical Report")

uploaded = st.file_uploader(
    "Choose a TXT file",
    type=["txt"]
)

report_text = ""

if uploaded is not None:
    report_text = uploaded.read().decode("utf-8")

report_text = st.text_area(
    "Medical Report Content",
    value=report_text,
    height=300,
    placeholder="Paste medical report text here..."
)

analyze = st.button(
    "🔍 Analyze Report",
    type="primary",
    use_container_width=True
)

if analyze and report_text.strip():

    specialty, confidence, probs, top_words = explain_prediction(report_text)

    st.subheader("🩺 Prediction Result")

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Predicted Specialty",
            specialty
        )

    with c2:
        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

    st.progress(confidence / 100)

    st.subheader("📊 Top Predictions")

    top_idx = np.argsort(probs)[::-1][:3]

    prob_df = pd.DataFrame({
        "Specialty": [
            label_encoder.inverse_transform([i])[0]
            for i in top_idx
        ],
        "Probability": [
            probs[i] * 100
            for i in top_idx
        ]
    })

    st.bar_chart(
        prob_df.set_index("Specialty")
    )

    st.subheader("🧠 Key Medical Terms")

    tags = ""

    for word, score in top_words[:15]:
        tags += f"""
        <span class="term-tag">
            {word}
        </span>
        """

    st.markdown(
        tags,
        unsafe_allow_html=True
    )

    chart_df = pd.DataFrame(
        top_words,
        columns=["Medical Term", "Importance"]
    )

    st.dataframe(chart_df, use_container_width=True)

    fig = px.bar(
        chart_df,
        x="Importance",
        y="Medical Term",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
        title="Attention-Based Medical Term Importance"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    terms = ", ".join(
    [w for w, _ in top_words[:5]]
    )

    st.markdown(f"""
    <div class="summary-card">

    <h3>📌 Medical Analysis Summary</h3>

    <b>Predicted Specialty:</b> {specialty}<br><br>

    <b>Confidence:</b> {confidence:.2f}%<br><br>

    <b>Important Medical Terms:</b> {terms}

    </div>
    """, unsafe_allow_html=True)