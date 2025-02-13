import gradio as gr
import os
import requests
from PyPDF2 import PdfReader
from docx import Document

# Watsonx.ai API details
WATSONX_API_KEY = os.getenv("XfyqbHqkZSatzDxeQzzEdQbfu-DP-_ihUvSSmrmIiTmT")
WATSONX_PROJECT_ID = os.getenv("289854e9-af72-4464-8bb2-4dedc59ad405")
WATSONX_ENDPOINT = "https://us-south.ml.cloud.ibm.com/ml/v1-beta/generation/text"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {WATSONX_API_KEY}",
    "ML-Instance-ID": WATSONX_PROJECT_ID
}

def generate_response(prompt: str, temperature: float, max_tokens: int):
    payload = {
        "model_id": "ibm/granite-13b-chat-v1",
        "inputs": prompt,
        "parameters": {"decoding_method": "sample", "temperature": temperature, "max_new_tokens": max_tokens}
    }
    response = requests.post(WATSONX_ENDPOINT, headers=HEADERS, json=payload)
    if response.status_code == 200:
        return response.json()["results"][0]["generated_text"]
    else:
        return f"Error: {response.text}"

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = "".join(page.extract_text() or "" for page in reader.pages)
    return text

def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    text = "\n".join(para.text for para in doc.paragraphs)
    return text

def analyze_resume(resume_text, job_description, with_job_description, temperature, max_tokens):
    if with_job_description:
        prompt = f"""
        Analyze this resume against the provided job description and provide:
        1. Match percentage
        2. Missing keywords
        3. Overall assessment
        4. Recommendations for improvement
        Job Description: {job_description}
        Resume: {resume_text}
        """
    else:
        prompt = f"""
        Analyze this resume and provide:
        1. Overall score
        2. Suggestions for improvement
        3. Key areas to focus on
        4. Recommendations with examples
        Resume: {resume_text}
        """
    return generate_response(prompt, temperature, max_tokens)

def process_resume(file):
    if file:
        file_type = file.name.split('.')[-1].lower()
        if file_type == 'pdf':
            return extract_text_from_pdf(file.name)
        elif file_type == 'docx':
            return extract_text_from_docx(file.name)
    return ""

with gr.Blocks() as demo:
    gr.Markdown("# 📄 ATS Resume Analyzer 📄")

    with gr.Tab("Resume Analyzer"):
        with_job_description = gr.Checkbox("Analyze with Job Description", value=True)
        job_description = gr.Textbox("Job Description", lines=5)
        resume_file = gr.File(label="Upload Resume (PDF or DOCX)")
        resume_content = gr.Textbox("Parsed Resume Content", lines=10)
        analyze_btn = gr.Button("Analyze Resume")
        output = gr.Markdown()

    temperature = gr.Slider(0, 1, step=0.1, value=0.5, label="Temperature")
    max_tokens = gr.Slider(50, 1024, step=1, value=1024, label="Max tokens")
    
    resume_file.upload(process_resume, resume_file, resume_content)
    analyze_btn.click(analyze_resume, [resume_content, job_description, with_job_description, temperature, max_tokens], output)

if __name__ == "__main__":
    demo.launch()
