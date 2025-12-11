<!-- ------------------------------------------------------------ -->
<!--                         PROJECT BANNER                        -->
<!-- ------------------------------------------------------------ -->

<p align="center">
  <img src="https://raw.githubusercontent.com/lavanya1402/HealthSenseAI/main/assets/banner.svg" width="95%" />
</p>

<h1 align="center">🌱 HealthSenseAI – Public Health Awareness Assistant</h1>

<p align="center">
<a href="https://aihealthsense.streamlit.app/"><img src="https://img.shields.io/badge/Live_Demo-Streamlit_App-FF4B4B?logo=streamlit&logoColor=white"></a>
<img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python">
<img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit">
<img src="https://img.shields.io/badge/RAG-FAISS-green">
<img src="https://img.shields.io/badge/LLM-Groq-orange?logo=groq">
<img src="https://img.shields.io/badge/License-MIT-yellow">
</p>

<p align="center">
  <a href="https://youtu.be/your_video_link_here">
    <img src="https://img.shields.io/badge/🎥_Video_Demo-Watch_Now-red?logo=youtube">
  </a>
</p>

---

## ⭐ Overview

**HealthSenseAI** is a multilingual health-education assistant designed for **India’s rural & semi-urban population**, powered by three core health documents:

- 🇮🇳 **Healthy Diet Guidelines**  
- 🩺 **Hypertension Screening Guidelines**  
- 👩‍🍼 **WHO Diabetes-Care Recommendations (Women)**  

Built using:

- ⚡ **Groq LLMs** (Llama / Mixtral)  
- 📚 **LangChain**  
- 🔍 **FAISS Vector Search**  
- 🖥️ **Streamlit UI**

It provides **safe, guideline-based** answers in **7 Indian languages**.

---

## ❤️ Why This Matters

India has:

- 65% population living in rural / semi-urban regions  
- Limited access to reliable health information  
- High rates of lifestyle diseases (BP, diabetes)  
- Huge linguistic diversity  

**HealthSenseAI solves this by offering:**

- Accurate ● multilingual ● guideline-verified health information  
- A tool ASHA workers can use for **mass family health education**  
- Zero-hallucination RAG for **trustworthy health awareness**

A single ASHA worker using this app can educate **hundreds of families**.

---

## ⚠️ Safety Disclaimer

HealthSenseAI is for **public health awareness ONLY**.

- ❌ Does NOT diagnose  
- ❌ Does NOT prescribe medicines  
- ❌ Cannot replace a medical professional  
- ⚠️ If symptoms are serious → consult a doctor immediately  

---

## 🧠 Core Features

### 🔍 Strict RAG Mode (Zero Hallucination)
- Answers ONLY from official guidelines  
- If a topic isn't covered → safe refusal  

### 🌐 Multilingual (7 Languages)
- English  
- Hindi  
- Marathi  
- Gujarati  
- Tamil  
- Telugu  
- Bengali  

### 🛡 Hard Safety Guardrails
- Blocks medical advice  
- Blocks prescriptions  
- Ensures responsible AI behavior  

---

# 🧱 Architecture Diagram (Mermaid)

```mermaid
flowchart TD

A[User Question] --> B[Language Detector]
B --> C[Safety Guardrails]

C -->|Allowed| D[RAG Pipeline]
C -->|Blocked| Z[Safe Refusal]

D --> E[FAISS Retriever]
E --> F[Relevant Guideline Chunks]

F --> G[Groq LLM (Llama/Mixtral)]
G --> H[Answer Generator]

H --> I[Streamlit UI Response]
```

---

# 🖼️ Banner (SVG auto-generated)

Save this file as: **assets/banner.svg**

```svg
<svg width="1200" height="250" xmlns="http://www.w3.org/2000/svg">
  <rect width="1200" height="250" fill="#e8f5e9"/>
  <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle"
        style="font-size:46px; font-family:Arial; fill:#2e7d32; font-weight:700;">
    🌱 HealthSenseAI — Public Health Awareness Assistant
  </text>
  <text x="50%" y="75%" dominant-baseline="middle" text-anchor="middle"
        style="font-size:22px; font-family:Arial; fill:#4a4a4a;">
    Multilingual | Guideline-Based | Safe | Powered by Groq + FAISS
  </text>
</svg>
```

---

# 🖼️ Social Preview Image (For GitHub SEO)

Create an image like below and save as:

```
/assets/social-preview.png
```

Upload in GitHub:

`Settings → General → Social Preview → Upload`

Preview:

```
+--------------------------------------------------------------+
| 🌱 HealthSenseAI                                             |
| Public Health Awareness Assistant                            |
|                                                              |
| • Multilingual (7 Languages)                                 |
| • Guideline-based RAG — Zero Hallucination                   |
| • Powered by Groq + FAISS + Streamlit                        |
|                                                              |
| https://aihealthsense.streamlit.app                          |
+--------------------------------------------------------------+
```

---

# 📁 Project Structure

```
HealthSenseAI/
│
├── data/
│   ├── raw/
│   │   ├── Healthy Diet.pdf
│   │   ├── Hypertension_full.pdf
│   │   ├── WHO recommendation on diabetes care for women.pdf
│   ├── processed/faiss_index/
│       ├── index.faiss
│       ├── index.pkl
│
├── src/
│   ├── app.py
│   ├── config.py
│   ├── rag_pipeline.py
│   ├── utils.py
│   ├── guards.py
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

# ⚙️ Installation

### 1️⃣ Create Virtual Environment
```
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate   # macOS/Linux
```

### 2️⃣ Install Dependencies
```
pip install -r requirements.txt
```

### 3️⃣ Add `.env`
```
APP_ENV=dev
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.1-8b-instant
DATA_RAW_DIR=data/raw
INDEX_DIR=data/processed/faiss_index
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

# ▶️ Run Locally

```
streamlit run src/app.py
```

---

# 📝 Sample Questions (7 Languages)

### English
- What foods are recommended in a healthy Indian diet?  
- How often should blood pressure be checked?  
- What diabetes screening test is recommended for women?

### Hindi  
- स्वस्थ आहार में क्या शामिल होना चाहिए?  
- ब्लड प्रेशर कितनी बार जांचना चाहिए?  
- महिलाओं के लिए डायबिटीज की कौन-सी जांच सुझाई गई है?

### Marathi  
- संतुलित आहारात काय असावे?  
- रक्तदाब किती वेळा तपासावा?  
- महिलांसाठी मधुमेह तपासणी काय?

### Gujarati  
- સંતુલિત આહારમાં શું લેવુ જોઈએ?  
- બ્લડ પ્રેશર ક્યારે તપાસવું?  
- મહિલાઓ માટે ડાયાબિટીસ સ્ક્રીનિંગ શું?

### Tamil  
- ஆரோக்கிய உணவில் என்ன சேர்க்க வேண்டும்?  
- இரத்த அழுத்தம் எவ்வளவு முறை பரிசோதிக்க வேண்டும்?

### Telugu  
- ఆరోగ్యకరమైన ఆహారంలో ఏమి ఉండాలి?  
- రక్తపోటు ఎప్పుడు పరీక్షించాలి?

### Bengali  
- স্বাস্থ্যকর খাদ্যে কী থাকা উচিত?  
- রক্তচাপ কতবার পরীক্ষা করা উচিত?

---

# 🏷 License

MIT License © 2025 Lavanya Srivastava


