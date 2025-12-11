# 🌿 HealthSenseAI – Public Health Awareness Assistant

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B.svg)]()
[![RAG](https://img.shields.io/badge/Architecture-RAG%20%2B%20FAISS-green.svg)]()
[![Groq](https://img.shields.io/badge/Powered%20by-Groq-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

### 🩺 AI-Powered, Multilingual, Guideline-Based Health Education

**Live Demo:** 👉 https://aihealthsense.streamlit.app/

⭐ Overview

HealthSenseAI is a multilingual AI assistant designed to improve public health awareness using official health guidelines like:

✅ Healthy Diet guidelines (India)

✅ Hypertension screening & management guidelines

✅ WHO recommendations on diabetes care for women

It uses Retrieval-Augmented Generation (RAG) with FAISS, Groq, and Streamlit to answer health questions safely, reliably, and in local languages.

⚠️ Important:
HealthSenseAI is an educational tool only.
It does not provide diagnosis, prescriptions, or treatment plans.

🧠 Core Features
🔍 1. Strict RAG (Zero-Hallucination Mode)

Answers are generated only from the uploaded guideline PDFs.

If no relevant guideline text is found, the assistant clearly says:

“The guideline does not provide information on this topic.”

If the FAISS index is unavailable or PDFs are unreadable, it reports:

“Guideline index unavailable.”

🌍 2. Multilingual Support (7 Languages)

The assistant responds automatically in the same language as the user’s question:

English (en)

Hindi (hi)

Marathi (mr)

Gujarati (gu)

Tamil (ta)

Telugu (te)

Bengali (bn)

Perfect for rural & semi-urban populations across India.

🛡️ 3. Hard Safety Guardrails

The assistant:

❌ Does not diagnose

❌ Does not prescribe medicines or doses

❌ Does not recommend treatments

✅ Encourages consulting qualified healthcare professionals

💻 4. Simple & Clean Web UI

Built with Streamlit, the app:

Runs in a browser (desktop or mobile)

Offers a smooth chat interface

Supports file uploads for guideline PDFs

🏗️ Architecture

LLM Backend: Groq (Llama/Mixtral models)

Orchestration: LangChain

Vector Store: FAISS

Embeddings: sentence-transformers/all-MiniLM-L6-v2

UI: Streamlit

Deployment: Streamlit Cloud

📂 Project Structure
HealthSenseAI/
│
├── data/
│   ├── raw/
│   │   ├── Healthy Diet.pdf
│   │   ├── Hypertension_full.pdf
│   │   ├── WHO recommendation on diabetes care for women.pdf
│   └── processed/
│       └── faiss_index/
│           ├── index.faiss
│           └── index.pkl
│
├── src/
│   ├── app.py              # Streamlit UI
│   ├── config.py           # Settings & LLM config
│   ├── rag_pipeline.py     # RAG pipeline (load / index / retrieve / answer)
│   ├── utils.py            # System prompts, language helpers
│   ├── guards.py           # Safety & guardrail filters
│
├── .env                    # Secrets (not committed)
├── requirements.txt
├── LICENSE                 # MIT License
└── README.md

⚙️ Setup & Local Run
1️⃣ Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# or
.venv\Scripts\activate      # Windows

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Set up .env

Create a .env file in the project root:

APP_ENV=dev
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
DATA_RAW_DIR=data/raw
INDEX_DIR=data/processed/faiss_index
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

4️⃣ Add guideline PDFs

Place your guideline PDFs in:

data/raw/


For example:

Healthy Diet.pdf

Hypertension_full.pdf

WHO recommendation on diabetes care for women.pdf

5️⃣ Run the app
streamlit run src/app.py


Open the local URL shown in your terminal (e.g., http://localhost:8501).

📸 Screenshots

Replace the placeholders below with real images from your deployed app.

🖼️ Home Screen
[ Add screenshot: main HealthSenseAI page with title + disclaimer ]


(Example: save as assets/home_screen.png and embed:)

![HealthSenseAI Home](assets/home_screen.png)

💬 Chat Interface (English)
[ Add screenshot: user asking about hypertension & AI answering from guidelines ]

🌐 Chat Interface (Hindi / Regional Language)
[ Add screenshot: user asking in Hindi/Marathi/Gujarati etc. ]

🏥 Why This App Matters (Especially in Developing Countries)

Millions lack access to specialist doctors.

Health guidelines exist (WHO/MoHFW), but are:

Long

Technical

Mostly in English

HealthSenseAI:

Makes guidelines searchable

Answers in simple language

Supports multiple Indian languages

Keeps strict safety (no diagnosis/prescription)

This makes it ideal for:

Rural health workers (ASHA / ANM)

Community health volunteers

NGOs in public health

Telemedicine support teams

Health awareness programs in schools & colleges

🧪 Testing Questions from All 3 Guideline PDFs

Use these sample questions to stress-test the RAG behaviour and show the importance of the app.

1️⃣ Healthy Diet Guidelines

English

What foods should be included in a balanced Indian diet?

How much sugar is recommended per day in a healthy diet?

What do the guidelines say about fat intake and heart health?

Hindi

संतुलित भारतीय आहार में कौन-कौन से खाद्य पदार्थ शामिल होने चाहिए?

रोज़ाना चीनी की कितनी मात्रा सुरक्षित मानी जाती है?

वसा (फैट) के बारे में दिशा-निर्देश क्या कहते हैं?

Gujarati

સ્વસ્થ ભારતીય આહારમાં કયા ખોરાકનો સમાવેશ કરવો જોઈએ?

એક દિવસમાં કેટલી ખાંડ લેવી યોગ્ય છે?

ચરબીના સેવન અંગે માર્ગદર્શિકા શું કહે છે?

2️⃣ Hypertension (High Blood Pressure) Guideline

English

What is the normal blood pressure range for adults as per the guideline?

How frequently should adults be screened for hypertension?

What lifestyle changes help in reducing the risk of high blood pressure?

Tamil

வழிகாட்டுதலின்படி பெரியவர்களுக்கான சாதாரண இரத்த அழுத்த அளவு எவ்வளவு?

பெரியவர்களின் ரத்த அழுத்தம் எத்தனை கால இடைவெளிக்கு ஒரு முறை பரிசோதிக்க வேண்டும்?

உயர் இரத்த அழுத்த அபாயத்தை குறைக்க எந்த வாழ்க்கை முறை மாற்றங்கள் உதவுகின்றன?

Telugu

మార్గదర్శకాల ప్రకారం పెద్దవారికి సాధారణ రక్తపోటు పరిధి ఎంత?

పెద్దవారి రక్తపోటును ఎంత వ్యవధికి ఒకసారి పరీక్షించాలి?

హైపర్‌టెన్షన్ ప్రమాదాన్ని తగ్గించడానికి ఏ జీవనశైలి మార్పులు సూచించబడ్డాయి?

3️⃣ WHO Recommendations on Diabetes Care for Women

English

What are the risk factors for diabetes in women according to the guideline?

How should diabetes be managed during pregnancy as per WHO?

What lifestyle measures are recommended to reduce diabetes risk in women?

Marathi

मार्गदर्शक तत्त्वांनुसार महिलांमध्ये मधुमेहाचे जोखीम घटक कोणते आहेत?

गर्भावस्थेदरम्यान मधुमेहाचे व्यवस्थापन कसे करावे, असे WHO काय सुचवते?

महिलांमध्ये मधुमेहाचा धोका कमी करण्यासाठी कोणते जीवनशैलीतील बदल सुचवले आहेत?

Bengali

নির্দেশিকা অনুযায়ী মহিলাদের মধ্যে ডায়াবেটিসের ঝুঁকির কারণগুলো কী কী?

WHO অনুযায়ী গর্ভাবস্থায় ডায়াবেটিস কীভাবে নিয়ন্ত্রণ করা উচিত?

মহিলাদের ডায়াবেটিসের ঝুঁকি কমাতে কী কী জীবনধারা পরিবর্তন সুপারিশ করা হয়েছে?

🔒 Ethical Disclaimer

HealthSenseAI is strictly for public health awareness & education.

It does not:

Diagnose any disease

Replace a doctor’s consultation

Suggest medicines, doses, or treatment plans

For any serious, persistent, or unclear symptoms, users should always consult:

Registered doctors

Local health centres

Government health helplines

🤝 Contributing

Contributions are welcome!

You can:

Add new guideline PDFs (e.g., TB, maternal health, anaemia, dengue)

Improve multilingual prompts and support

Enhance UI/UX for low-literacy users

Add voice input/output

📜 License

This project is licensed under the MIT License – see the LICENSE
 file for details.

💛 Author

Lavanya Srivastava
AI Educator • Public Health Awareness Enthusiast • Agentic AI Developer

Deployed App: https://aihealthsense.streamlit.app/

GitHub: https://github.com/lavanya1402

LinkedIn: (add your profile link here)
