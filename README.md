
# 🤖 AI vs Student Text Classifier

This project fine-tunes a BERT-based Large Language Model to classify whether a given paragraph was written by a human student or generated by AI. It includes model training, evaluation, error analysis, and a fully functional Streamlit demo.

---

## 🚀 Live Demo

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-green?logo=streamlit)](https://llm-text-detector-ai-vs-student.streamlit.app)

🔗 **Hugging Face Model**: [sohnirais/llm_detector](https://huggingface.co/sohnirais/llm_detector)

---

## 📌 Project Overview

- **Model**: `bert-base-uncased` fine-tuned for binary classification (`AI` vs. `Student`)
- **Dataset**: [Kaggle LLM Text Detection Dataset](https://www.kaggle.com/datasets/prajwaldongre/llm-detect-ai-generated-vs-student-generated-text)
- **Goal**: Develop a classifier that detects whether text is machine-generated or human-written in academic settings.

---

## 📁 Folder Structure

```
llm-text-detector-ai-vs-student/
│
├── streamlit_llm.py        # Streamlit UI for text classification
├── LLM.ipynb               # Fine-tuning + Evaluation notebook
├── llm-detector/           # Model folder (config.json, tokenizer, pytorch_model.bin)
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
└── assets/                 # Screenshots and plots
```

---

## 🧪 Model Performance

| Metric     | Score    |
|------------|----------|
| Accuracy   | 100%     |
| Precision  | 100%     |
| Recall     | 100%     |
| F1 Score   | 100%     |

🔍 **Note**: High scores due to clean, labeled dataset and clear AI vs. student writing patterns.

---

## 📊 Training Logs & Visuals

Training and evaluation visualized using [Weights & Biases](https://wandb.ai/):

- Loss vs Epoch
- Learning Rate Scheduling
- Gradient Norm

Visuals included in the `assets/` folder and report.

---

## 🔬 Inference Pipeline (Streamlit UI)

The app allows users to paste a paragraph and instantly get:

- AI or Student classification
- Confidence score
- Live interaction from browser

---

## 🛠️ Setup Instructions

### 1. Clone the Repo
```bash
git clone https://github.com/raiss15/llm-text-detector-ai-vs-student.git
cd llm-text-detector-ai-vs-student
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Streamlit App
```bash
streamlit run streamlit_llm.py
```

---

## 🔗 References

- [Kaggle Dataset](https://www.kaggle.com/datasets/prajwaldongre/llm-detect-ai-generated-vs-student-generated-text)
- [Hugging Face Model Repo](https://huggingface.co/sohnirais/llm_detector)
- [BERT base uncased](https://huggingface.co/bert-base-uncased)
- [Weights & Biases](https://wandb.ai/)
- [Streamlit](https://streamlit.io/)

---

## 👩‍💻 Author

**Sohni Rais**  
Graduate Student, Northeastern University  
[LinkedIn](https://www.linkedin.com/in/sohnirais) | [GitHub](https://github.com/raiss15)

---
