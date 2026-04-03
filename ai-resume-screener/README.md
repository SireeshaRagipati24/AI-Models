# 🤖 AI Resume Screener & Interview Coach

An intelligent, high-precision ATS tool that analyzes resumes against Job Descriptions using **Hybrid NLP (Cosine Similarity)** and **Semantic AI (Gemini)**. It provides deep analysis, skill gap detection, and dynamic interview preparation.

---

## ✨ Features
- 📄 **PDF Extraction** — High-accuracy text parsing using PyMuPDF.
- 🧠 **Hybrid Scoring** — Combines **TF-IDF Vectorization** with **LLM Semantic Analysis**.
- 🚀 **Dynamic Interview Prep** — Automatically generates 3 tough technical questions based on resume gaps.
- 🎯 **Perfect Fit Guide** — Specific advice on exactly what keywords or phrases to add to reach a 100% match.
- 📊 **Visual Analytics** — Animated match bars, A/B/C/D grading, and strength highlights.
- 🎨 **Modern Dark UI** — Premium purple-gradient aesthetic with glassmorphism effects.

---

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Backend** | Python, Flask |
| **AI Model** | Google Gemini 1.5 Flash |
| **NLP Logic** | Cosine Similarity, Scikit-learn (Custom), Counter |
| **PDF Parsing** | PyMuPDF (fitz) |
| **Frontend** | HTML5, CSS3 (Modern Gradients), JavaScript (Async/Fetch) |
| **Deployment** | Render / Gunicorn |

---

## 📁 Project Structure

```text
ai-resume-screener/
├── templates/
│   └── index.html        # Modern Responsive Frontend
├── app.py                # Flask API & Route Handling
├── parser.py             # PDF Text Extraction Engine
├── scorer.py             # Mathematical Keyword Matching Logic
├── gemini_scorer.py      # AI Semantic Analysis & Question Generator

---

## 🚀 Run Locally

**Step 1 — Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/ai-resume-screener.git
cd ai-resume-screener
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Run**
```bash
python app.py
```

**Step 4 — Open browser**

---

---

## 📸 Demo

### Upload Screen
- Clean drag & drop interface
- Paste any real job description

### Results Screen
- **Grade A** → 75%+ match → *"Excellent match — strong candidate!"*
- **Grade B** → 55–74% → *"Good match — worth pursuing"*
- **Grade C** → 35–54% → *"Partial match — some gaps to address"*
- **Grade D** → below 35% → *"Consider tailoring your resume"*

---

## 📦 Dependencies
- flask
- pymupdf
- python-dotenv
- gunicorn

---

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🧠 NLP Logic — How Scoring Works

### Keyword Match (TF-IDF Cosine Similarity)
Converts both the resume and job description into word frequency vectors, then computes the cosine angle between them. Higher overlap = higher score.

### Semantic Score
Extracts meaningful keywords (excluding stopwords), compares them against JD requirements, and scores based on contextual relevance — not just exact word matches.

### Skill Gap Detection
Checks for 20+ in-demand tech skills (Python, SQL, Flask, TensorFlow, Docker, AWS, etc.) and tells you exactly which ones are present and which are missing.

---

## 👩‍💻 Built By

**Sireesha Ragipati**  
Associate Data Scientist | NLP & GenAI Enthusiast  
Hyderabad, India

---

## 🌐 Live Demo
You can try the live version of this AI Resume Screener here:
### [👉 Click Here to View Live Project](https://ai-model-screener.onrender.com)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*If this project helped you, consider giving it a ⭐ on GitHub!*
├── requirements.txt      # Project Dependencies
└── .env                  # Environment Variables (API Keys)
