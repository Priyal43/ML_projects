from flask import Blueprint, render_template, request, flash
from app.utils import extract_information, compare_resume_and_jd, parse_resume  # Ensure parse_resume is imported

# Define a blueprint
main = Blueprint("main", __name__, template_folder="templates")

@main.route("/")
def index():
    return render_template("index.html")

@main.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        flash("No file part in the request. Please upload your resume.")
        return render_template("index.html")

    file = request.files["file"]
    jd_text = request.form.get("job_description", "").strip()

    if file.filename == "":
        flash("No file selected. Please choose a resume to upload.")
        return render_template("index.html")

    if file and file.filename.lower().endswith(".pdf"):
        try:
            # Parse the resume text using parse_resume
            resume_text = parse_resume(file)  # Use parse_resume here
            
            # Extract information from the resume and job description
            resume_info = extract_information(resume_text)
            jd_info = extract_information(jd_text, is_job_description=True)

            # Compare resume and job description information
            feedback = compare_resume_and_jd(resume_info, jd_info)

            # Render the results page with feedback
            return render_template("results.html", feedback=feedback)
        except Exception as e:
            flash(f"Error processing the file: {e}")
            return render_template("index.html")
    else:
        flash("Invalid file format. Please upload a PDF resume.")
        return render_template("index.html")
