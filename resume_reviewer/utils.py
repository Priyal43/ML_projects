import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch
import json
import pdfplumber
import pytesseract
from pdf2image import convert_from_path

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Load the BERT model and tokenizer
bert_model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
model = AutoModel.from_pretrained(bert_model_name)

def parse_resume(file):
    """
    Parse the resume file and extract text using pdfplumber or OCR as a fallback.
    """
    try:
        with pdfplumber.open(file) as pdf:
            text = "".join(page.extract_text() for page in pdf.pages)
        
        if not text.strip():
            images = convert_from_path(file)
            text = " ".join(pytesseract.image_to_string(image) for image in images)
        
        return text
    except Exception as e:
        raise Exception(f"Error processing the file: {e}")

# Predefined skills dataset
with open("skills_list.json", "r") as file:
    COMMON_SKILLS = set(json.load(file))  # A list of predefined skills

def extract_skills_dynamic(text):
    """
    Dynamically extract potential skills from text using predefined skills and contextual filters.
    """
    doc = nlp(text.lower())
    skills = set()

    for chunk in doc.noun_chunks:
        token_text = chunk.text.strip()
        if token_text in COMMON_SKILLS:
            skills.add(token_text)
    
    for token in doc:
        if token.text in COMMON_SKILLS and not token.is_stop:
            skills.add(token.text)

    return list(skills)

def bert_similarity(text1, text2):
    """
    Compute semantic similarity between two texts using BERT embeddings.
    """
    with torch.no_grad():
        inputs1 = tokenizer(text1, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
        inputs2 = tokenizer(text2, return_tensors="pt", truncation=True, max_length=512, padding="max_length")

        embeddings1 = model(**inputs1).last_hidden_state.mean(dim=1)
        embeddings2 = model(**inputs2).last_hidden_state.mean(dim=1)

        similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2).item()
        return similarity

def extract_information(text, is_job_description=False):
    """
    Extract structured information from the given text dynamically.
    """
    skills = extract_skills_dynamic(text)
    education = []
    experience = []

    if not is_job_description:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "ORG":
                experience.append(ent.text)
            elif ent.label_ == "GPE":
                education.append(ent.text)
        
        degree_patterns = r"(Bachelor|B\.?Sc|B\.?Eng|Master|M\.?Sc|PhD|Diploma|Certification)"
        education += re.findall(degree_patterns, text, re.IGNORECASE)

    education = list(filter(lambda x: len(x.split()) > 1, education))
    experience = list(filter(lambda x: len(x.split()) > 1, experience))

    return {
        "skills": skills,
        "education": list(set(education)),
        "experience": list(set(experience))
    }

def compare_resume_and_jd(resume_info, jd_info):
    """
    Compare extracted resume information with job description information using BERT for semantic similarity.
    """
    missing_skills = set(jd_info["skills"]) - set(resume_info["skills"])
    matched_skills = set(resume_info["skills"]) & set(jd_info["skills"])
    missing_experience = set(jd_info["experience"]) - set(resume_info["experience"])

    feedback = ""

    if missing_skills:
        feedback += f"Consider learning these skills to match the JD better:\n- {', '.join(missing_skills)}\n"
    else:
        feedback += "You already possess all the required skills for this role!\n"

    if matched_skills:
        feedback += f"\nYour resume highlights these relevant skills:\n- {', '.join(matched_skills)}\n"

    if missing_experience:
        feedback += f"\nYou may need experience in:\n- {', '.join(missing_experience)}\n"
    else:
        feedback += "\nYour experience aligns well with the job requirements.\n"

    # Semantic similarity using BERT
    # Convert lists to space-separated strings
    resume_text = " ".join([
        " ".join(resume_info["skills"]),
        " ".join(resume_info["education"]),
        " ".join(resume_info["experience"])
    ])

    jd_text = " ".join([
        " ".join(jd_info["skills"]),
        " ".join(jd_info["education"]),
        " ".join(jd_info["experience"])
    ])
    
    semantic_similarity = bert_similarity(resume_text, jd_text)

    feedback += f"\nSemantic similarity with job description: {semantic_similarity:.2f}\n"
    if semantic_similarity < 0.5:
        feedback += "Consider revising your resume for better alignment with the job description."

    layout_feedback = []
    if not resume_info["skills"]:
        layout_feedback.append("Include a Skills section.")
    if not resume_info["education"]:
        layout_feedback.append("Include an Education section.")
    if not resume_info["experience"]:
        layout_feedback.append("Include an Experience section.")
    if not layout_feedback:
        layout_feedback.append("Your layout is well-structured.")

    feedback += "\nLayout Suggestions:\n- " + "\n- ".join(layout_feedback)

    return feedback
