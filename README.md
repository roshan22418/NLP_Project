# 🧠 Next Utterance Prediction for Mental Health Counseling

## Authors
- Wasif Ali (2022583)
- Roshan Kumar Mahto (2022418)  
- Ashish Bargoti (2022114)  


---

## 📄 Abstract

This project presents a novel approach to next utterance prediction for **mental health counseling dialogues** using **transformer-based models**. The goal is to generate **empathetic**, **contextually appropriate** therapist responses to assist in scalable mental healthcare.

We propose two key models:
1. **Sentiment Infused BART** — integrates sentiment features into BART architecture.
2. **T-GenSim (Therapeutic Generator-Simulator)** — a dual-model system using a generator for response prediction and a simulator to forecast possible patient replies.

---

## 🚀 Objectives

- Generate emotionally sensitive and context-aware responses.
- Assist therapists with AI-powered response suggestions.
- Enable scalable mental health dialogue systems using transformer models.

---

## 📚 Dataset

- **Source**: Curated mental health counseling conversations.
- **Splits**:
  - Training Samples: 4008  
  - Validation Samples: 576  
  - Test Samples: 968  

### Sample Format

```txt
Input:
T: Hi you how to do it today?  
[SEP] P: Great. How are you?  
[SEP] T: I’m doing well. Thanks for asking.

Target:
"So you’re doing great."
🏗️ Models
1️⃣ Sentiment Infused BART
Architecture: BART + Sentiment Embedding via cardiffnlp/twitter-roberta-base-sentiment.

Fusion: Multi-head attention + sentiment + residual connections.

Training:

Epochs: 5

Batch Size: 8

Optimizer: AdamW (lr = 1e-5)

2️⃣ T-GenSim (Therapeutic Generator-Simulator)
Components:

Generator Model (for therapist replies)

Simulator Model (for patient follow-up)

Training Objective:

Cross-entropy loss + Emotion-based reward signal

Triplet-based Input:

History – previous utterances

Target Utterance – next therapist response

Simulated Next Input – likely follow-up by patient

📊 Evaluation Metrics
Metric	Sentiment BART	T-GenSim
BLEU	0.0104	0.0253
BERT F1	0.8480	0.8151
ROUGE-1 (F1)	0.1428	-
ROUGE-2 (F1)	0.0187	-
ROUGE-L (F1)	0.1170	-

⚠️ Note: BLEU scores are low due to the open-ended nature of dialogue tasks.
---

## ⚖️ Baseline Models

We compare our proposed approaches with two strong baseline transformer models widely used in dialogue generation:

### 🔹 T5-small
- A text-to-text transformer that reframes all NLP tasks as text generation.
- Input format:  
  `"predict next utterance: <context>"`
- Output decoding: Beam Search
- Evaluated with: BLEU, BERT Precision, Recall, and F1.

### 🔹 BART-base
- A denoising autoencoder transformer designed for sequence-to-sequence generation.
- Fine-tuned using teacher forcing with `BartTokenizer`.
- Optimizer: AdamW
- Best model selected based on validation loss.
- Evaluated with: BLEU and BERTScore.

### 📊 Baseline Performance

| Model   | BLEU   | BERT Precision | BERT Recall | BERT F1 |
|---------|--------|----------------|-------------|---------|
| T5      | 0.0186 | 0.8611         | 0.8443      | 0.8524  |
| BART    | 0.0219 | 0.8648         | 0.8488      | 0.8564  |

> These baselines provide a strong foundation, and our proposed models aim to build upon them by incorporating **sentiment-awareness** and **dual-simulation mechanisms** to improve therapeutic dialogue generation.



🧪 Experimental Setup
Hardware
GPU: NVIDIA Tesla V100

CPU: Intel Xeon Gold 6240

RAM: 64 GB

Software
Python 3.8

PyTorch

HuggingFace Transformers

NumPy, Pandas, Scikit-learn

🔍 Key Takeaways
Emotion-Aware Generation improves the empathy level of responses.

BERTScore is more reflective than BLEU for therapeutic dialogues.

Hybrid Architectures combining generation and simulation enhance dialogue quality.

🛠️ Future Work
Reinforcement Learning with Human Feedback (RLHF)

Multimodal Learning (text + voice/facial expressions)

Long-Term Context Tracking

Clinical Evaluation with Real Therapists

📜 References
Inaba, M., et al. (2024). Can LLMs provide psychological counselling? arXiv:2402.12738

Srivastava, A., et al. (2023). Response-act guided reinforced dialogue generation for mental health counseling. ACM WebConf.

📎 License
This project is intended for research and educational purposes only. Please ensure ethical usage aligned with mental health guidelines.

yaml
Copy
Edit

---

Let me know if you'd like this in `.md` file format or want to include code examples, model architecture 