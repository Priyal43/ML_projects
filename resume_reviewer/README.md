# AI-Powered Resume Reviewer
This project is an AI-driven web application designed to help job seekers refine their resumes. By leveraging advanced models like BERT and spaCy, the application provides actionable insights and tailored recommendations to optimize resumes for specific job descriptions.

## Features
**Skill Extraction:** Dynamically detects skills from resumes using a predefined dataset and NLP-based contextual filters.  
**Semantic Comparison:** Utilizes BERT for meaningful comparisons between resumes and job descriptions, providing precise feedback.  
**Education and Experience Analysis:** Extracts and highlights qualifications, institutions, and professional experiences.  
**Feedback and Suggestions:** Offers layout improvement tips and targeted recommendations to enhance resume alignment with job requirements.  

## Technology Stack
**Backend:** Flask (Python)  
**AI/NLP Models:** BERT, spaCy, scikit-learn  
**PDF Parsing:** pdfplumber, Tesseract OCR, pdf2image  
**Frontend:** HTML, CSS  

## Installation
Follow the steps below to set up the project on your local machine:

### Prerequisites
**Python 3.8+ installed on your system.**

**Poppler for PDF processing:**
-  macOS: brew install poppler
-  Windows: Install Poppler for Windows

**Tesseract OCR:**
-  macOS: brew install tesseract
-  Windows: Install Tesseract OCR for Windows

### Quick Setup
**Clone this repository:**
- git clone https://github.com/yourusername/resume-reviewer.git  
- cd resume-reviewer  

**Install the required dependencies:**
- pip install -r requirements.txt  

**Run the application:**
- python app.py  
- Open your browser and navigate to: http://127.0.0.1:8000

## Usage
1. Launch the application and upload your resume in PDF format.
2. Optionally, paste a job description to compare.
3. Submit the form to receive detailed feedback.

## File Structure
 
```
resume-reviewer/  
├── app/  
│   ├── __init__.py         
│   ├── routes.py           
│   ├── utils.py            
│   ├── templates/  
│   │   ├── index.html      
│   │   ├── results.html    
│   └── static/  
│       └── style.css       
├── skills_list.json        
├── requirements.txt       
├── Procfile                
└── README.md   
```         
