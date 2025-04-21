import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# ✅ Set your Hugging Face repo
model_repo = "sohnirais/llm_detector"

# ✅ Configure Streamlit page
st.set_page_config(page_title="AI vs Student Text Classifier", layout="wide")

# ✅ Load model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_repo)
model = AutoModelForSequenceClassification.from_pretrained(model_repo)

# ✅ Sidebar
with st.sidebar:
    st.title("📘 About")
    st.markdown(
        "This app classifies whether a paragraph is written by an AI or a student using a fine-tuned BERT model."
    )
    st.markdown(
        "🔍 **Model**: `bert-base-uncased` fine-tuned on the "
        "[LLM dataset](https://www.kaggle.com/datasets/prajwaldongre/llm-detect-ai-generated-vs-student-generated-text)"
    )
    st.markdown("👩‍💻 Built by Sohni Rais")

# ✅ App Title
st.title("📚 AI vs Student Text Classifier")
st.markdown("Enter a paragraph to check if it was written by an AI or a student.")

# ✅ Text input
text_input = st.text_area(
    "📝 Paste your paragraph below:",
    placeholder="E.g., The growing impact of AI in education has sparked discussions...",
    height=200
)

# ✅ Classification logic
if st.button("🚀 Classify"):
    if text_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        with st.spinner("Classifying..."):
            inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits).item()
            probs = F.softmax(outputs.logits, dim=1)
            confidence = probs[0][prediction].item()
            label = "🧑‍🎓 Student" if prediction == 0 else "🤖 AI"

            st.markdown(
                f"<div style='background-color:#e7f3fe;padding:16px;border-radius:10px;margin-top:10px;'>"
                f"<h4 style='color:#084298;'>Prediction: <strong>{label}</strong></h4>"
                f"<p style='color:#084298;'>Confidence: <strong>{confidence:.2%}</strong></p>"
                "</div>",
                unsafe_allow_html=True
            )

# ✅ Footer
st.markdown("---")
st.markdown(
    "<center><small>© 2025 Sohni Rais | AI vs Student Text Classifier</small></center>",
    unsafe_allow_html=True
)
