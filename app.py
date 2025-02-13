import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PyPDF2 import PdfReader
from docx import Document

# Load DeepSeek Model & Tokenizer
MODEL_NAME = "deepseek-ai/deepseek-llm-7b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def generate_response(prompt, max_tokens=512, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    output = model.generate(**inputs, max_length=max_tokens, temperature=temperature)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def analyze_resume(resume_text, job_description, with_job_description):
    if with_job_description:
        prompt = f"""
        Analyze the following resume against the provided job description. 
        Provide: 
        1. Match percentage 
        2. Missing skills 
        3. Key recommendations
        \n\nJob Description:\n{job_description}\n\nResume:\n{resume_text}
        """
    else:
        prompt = f"""
        Analyze the following resume and provide: 
        1. Overall strengths and weaknesses 
        2. Key skills identified 
        3. Suggestions for improvement\n\nResume:\n{resume_text}
        """
    return generate_response(prompt)

def process_resume(file):
    if file is not None:
        file_type = file.name.split('.')[-1].lower()
        if file_type == 'pdf':
            return extract_text_from_pdf(file.name)
        elif file_type == 'docx':
            return extract_text_from_docx(file.name)
    return ""

with gr.Blocks() as demo:
    gr.Markdown("## 📄 Resume Analyzer using DeepSeek LLM 📄")
    
    with gr.Row():
        resume_file = gr.File(label="Upload Resume (PDF or DOCX)")
        resume_content = gr.Textbox(label="Extracted Resume Content", lines=10)
    
    with_job_description = gr.Checkbox(label="Analyze with Job Description", value=True)
    job_description = gr.Textbox(label="Job Description (Optional)", lines=5)
    
    analyze_btn = gr.Button("Analyze Resume")
    output = gr.Markdown()
    
    resume_file.upload(process_resume, resume_file, resume_content)
    analyze_btn.click(analyze_resume, inputs=[resume_content, job_description, with_job_description], outputs=[output])

demo.launch()
