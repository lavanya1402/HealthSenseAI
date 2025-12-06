![HealthSenseAI Banner](assets/banner.png)

# 🩺 HealthSenseAI  
**AI Assistant for Public Health Awareness & Early Risk Guidance**

---

## 🏷️ At a Glance

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)
![Groq](https://img.shields.io/badge/LLM-Groq%20Llama3.1%208B-orange)
![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-2E77BB.svg)
![HuggingFace](https://img.shields.io/badge/Embeddings-HuggingFace-yellow.svg)
![RAG](https://img.shields.io/badge/Architecture-RAG-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 1. Executive Summary

**HealthSenseAI** is a multilingual, guideline-aware public health assistant built using **Generative AI + Retrieval-Augmented Generation (RAG)**.

It is designed to:

- Help citizens understand **symptoms, risks, prevention, and screenings** using credible public health content.
- Reduce the burden on hospitals and clinics for **non-emergency informational queries**.
- Demonstrate how **LLMs, vector search, and guardrails** can be combined to deliver safe, explainable AI in healthcare settings.

The solution retrieves content from uploaded **WHO / CDC / MoHFW** public-health guidelines, indexes them with FAISS, and responds using **Groq’s high-performance Llama 3.1 models**, wrapped in a clean Streamlit UI.

> **Important:** HealthSenseAI is an educational tool. It does **not** provide diagnosis, prescriptions, or clinical decision support.

---

## 2. Business Problem

Healthcare systems worldwide face similar challenges:

- ❌ Patients search on Google and receive **inconsistent, unverified information**.  
- ❌ Hospitals and helplines are overwhelmed with **non-urgent questions**.  
- ❌ Public health guidelines are available but **hard to navigate** and not always localized.  
- ❌ Reliable content is often **available only in English**, limiting accessibility.

**Organizations affected:**

- Public hospitals and clinics  
- Government health departments  
- NGOs and community health programs  
- Health insurers and wellness platforms  
- Telemedicine and health-app providers  

---

## 3. Solution Overview

**HealthSenseAI** addresses these challenges by providing:

- ✅ An **AI assistant** that explains symptoms, risk factors, screenings, and lifestyle modifications in simple language.
- ✅ **Multilingual support** (English, Hindi, Marathi) to improve reach and accessibility.
- ✅ **RAG-based retrieval** from trusted guideline PDFs, rather than relying purely on the model’s memory.
- ✅ **Guardrails** that prevent unsafe behavior (no diagnosis, no prescriptions, explicit disclaimers).

The result is a **safe, explainable, and configurable** AI layer that can sit in front of existing health information systems or be used independently for public health awareness.

---

## 4. High-Level Architecture

### 4.1 Teal & White Architecture Diagram (for slides / docs)

> Place a PNG diagram at:  
> `assets/architecture_teal_white.png`

```md
![HealthSenseAI Architecture](assets/architecture_teal_white.png)
