import math

def get_tfidf_score(resume_text, jd_text):
    def tokenize(text):
        return set(text.lower().split())
    
    resume_words = tokenize(resume_text)
    jd_words = tokenize(jd_text)
    
    if not jd_words:
        return 0.0
    
    common = resume_words & jd_words
    score = len(common) / math.sqrt(len(resume_words) * len(jd_words))
    return round(score * 100, 2)

def get_skill_gap(resume_text, jd_text):
    skills_list = [
        "python", "sql", "machine learning", "nlp", "flask", "tensorflow",
        "keras", "scikit-learn", "pandas", "numpy", "git", "docker",
        "aws", "react", "postgresql", "mysql", "deep learning", "llm",
        "generative ai", "gemini", "power bi", "tableau", "xgboost"
    ]
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()
    jd_skills = set(s for s in skills_list if s in jd_lower)
    resume_skills = set(s for s in skills_list if s in resume_lower)
    matched = list(jd_skills & resume_skills)
    missing = list(jd_skills - resume_skills)
    return matched, missing