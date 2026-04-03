from flask import Flask, request, jsonify, render_template
from parser import extract_resume_text
from scorer import get_tfidf_score, get_skill_gap
from gemini_scorer import get_gemini_score
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/screen", methods=["POST"])
def screen_resume():
    try:
        if "resume" not in request.files:
            return jsonify({"error": "Resume file required"}), 400
        if "job_description" not in request.form:
            return jsonify({"error": "Job description required"}), 400

        resume_file = request.files["resume"]
        jd_text = request.form["job_description"]

        file_path = os.path.join(UPLOAD_FOLDER, resume_file.filename)
        resume_file.save(file_path)

        resume_text = extract_resume_text(file_path)
        tfidf_score = get_tfidf_score(resume_text, jd_text)
        matched_skills, missing_skills = get_skill_gap(resume_text, jd_text)
        gemini_result = get_gemini_score(resume_text, jd_text)

        return jsonify({
            "candidate": resume_file.filename,
            "tfidf_match_score": tfidf_score,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "gemini_semantic_score": gemini_result["semantic_score"],
            "experience_match": gemini_result["experience_match"],
            "summary": gemini_result["summary"],
            "strengths": gemini_result["strengths"],
            "improvements": gemini_result["improvements"]
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)