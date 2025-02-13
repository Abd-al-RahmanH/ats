import gradio as gr 
import os
from groq import Groq
from PyPDF2 import PdfReader
from docx import Document

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

CSS = """
.duplicate-button { 
    margin: auto !important; 
    color: white !important; 
    background: black !important; 
    border-radius: 100vh !important;
}
h3, p, h1 { 
    text-align: center; 
    color: white;
}
footer { 
    text-align: center; 
    padding: 10px; 
    width: 100%; 
    background-color: rgba(240, 240, 240, 0.8); 
    z-index: 1000; 
    position: relative; 
    margin-top: 10px; 
    color: black;
}
"""

FOOTER_TEXT = """
<footer>
    <p>If you enjoyed the functionality of the app, please leave a like!<br>
    Check out more on <a href="https://www.linkedin.com/in/girish-wangikar/" target="_blank">LinkedIn</a> | 
    <a href="https://girishwangikar.github.io/Girish_Wangikar_Portfolio.github.io/" target="_blank">Portfolio</a></p>
</footer>
"""

RESUME_ANALYZER_INSTRUCTIONS = """
<div style="background-color: #000000; color: #ffffff; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
    <p><strong>Instructions:</strong></p>
    <ul>
        <li>Upload your resume (PDF or DOCX) in the file upload area.</li>
        <li>If you want to analyze your resume against a specific job description, keep the checkbox checked and enter the job description in the text box.</li>
        <li>If you want a general resume analysis without a job description, uncheck the "Analyze with Job Description" box.</li>
        <li>Click "Analyze Resume" to get your results.</li>
    </ul>
</div>
"""

COVER_LETTER_INSTRUCTIONS = """
<div style="background-color: #000000; color: #ffffff; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
    <p><strong>Instructions for Cover Letter Generation:</strong></p>
    <ol>
        <li>First, go to the "Resume Analyzer" tab.</li>
        <li>Upload your resume and enter the job description there.</li>
        <li>Then, come back to this tab and click "Generate Cover Letter".</li>
    </ol>
</div>
"""

INTERVIEW_QUESTIONS_INSTRUCTIONS = """
<div style="background-color: #000000; color: #ffffff; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
    <p><strong>Instructions for Interview Questions Generation:</strong></p>
    <p>Enter the job description in the text box below and click "Generate Interview Questions".</p>
</div>
"""

# Also update the disclaimer styles to match
COVER_LETTER_DISCLAIMER = """
<p style="font-style: italic; color: #cccccc; background-color: #000000; padding: 10px; border-radius: 5px;">
Disclaimer: This cover letter is generated based on the provided job description and resume. 
It should be carefully reviewed and tailored to your specific needs and the company's requirements before use.
</p>
"""

INTERVIEW_QUESTIONS_DISCLAIMER = """
<p style="font-style: italic; color: #cccccc; background-color: #000000; padding: 10px; border-radius: 5px;">
Disclaimer: These interview questions are generated based on the provided job description. 
They should be reviewed and adjusted to better fit the specific role, company culture, and interview process.
</p>
"""

TITLE = "<h1>📄 ATS Resume Analyzer 📄</h1>"
PLACEHOLDER = "Chat with AI about your resume and job descriptions..."

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

def generate_response(message: str, system_prompt: str, temperature: float, max_tokens: int):
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message}
    ]

    response = client.chat.completions.create(
        model="llama-3.1-8B-Instant",
        messages=conversation,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False
    )

    return response.choices[0].message.content

    
def analyze_resume_with_job_description(resume_text, job_description, temperature, max_tokens):
    prompt = f"""
    Please analyze the following resume in the context of the job description provided. Strictly check every single line in the job description and analyze the resume for exact matches. Maintain high ATS standards and give scores only to the correct matches. Focus on missing core skills and soft skills. Provide the following details:
    1. The match percentage of the resume to the job description.
    2. A list of missing keywords.
    3. Final thoughts on the resume's overall match with the job description in 3 lines.
    4. Recommendations on how to add the missing keywords and improve the resume in 3-4 points with examples.
    Job Description: {job_description}
    Resume: {resume_text}
    """
    return generate_response(prompt, "You are an expert ATS resume analyzer.", temperature, max_tokens)

def analyze_resume_without_job_description(resume_text, temperature, max_tokens):
    prompt = f"""
    Please analyze the following resume without a specific job description. Provide the following details:
    1. An overall score out of 10 for the resume.
    2. Suggestions for improvements based on the following criteria:
       - Impact (quantification, repetition, verb usage, tenses, responsibilities, spelling & consistency)
       - Brevity (length, bullet points, filler words)
       - Style (buzzwords, dates, contact details, personal pronouns, active voice, consistency)
       - Sections (summary, education, skills, unnecessary sections)
    3. A cumulative assessment of all the above fields.
    4. Recommendations for improving the resume in 3-4 points with examples.

    Resume: {resume_text}
    """
    return generate_response(prompt, "You are an expert ATS resume analyzer.", temperature, max_tokens)


def analyze_resume(resume_text, job_description, with_job_description, temperature, max_tokens):
    if with_job_description:
        return analyze_resume_with_job_description(resume_text, job_description, temperature, max_tokens)
    else:
        return analyze_resume_without_job_description(resume_text, temperature, max_tokens)



def rephrase_text(text, temperature, max_tokens):
    prompt = f"""
    Please rephrase the following text according to ATS standards, including quantifiable measures and improvements where possible. Maintain precise and concise points which will pass ATS screening:
    Original Text: {text}
    """
    return generate_response(prompt, "You are an expert in rephrasing content for ATS optimization.", temperature, max_tokens)

def clear_conversation():
    return [], None


def generate_cover_letter(resume_text, job_description, temperature, max_tokens):
    prompt = f"""
    Using the provided resume and job description, create a compelling cover letter. The cover letter should:
    1. Be tailored to the specific job and company.
    2. Highlight relevant skills and experiences from the resume.
    3. Show enthusiasm for the role and company.
    4. Be professional and concise (about 250-300 words).

    Resume: {resume_text}
    Job Description: {job_description}
    """
    return generate_response(prompt, "You are an expert in writing tailored cover letters.", temperature, max_tokens)

def generate_interview_questions(job_description, temperature, max_tokens):
    prompt = f"""
    Based on the following job description, generate a list of 10 probable interview questions. Include a mix of:
    1. Role-specific technical questions (if applicable)
    2. Behavioral questions related to the required skills
    3. Questions about the candidate's experience and background
    4. Questions to assess cultural fit

    Ensure the questions are tailored to the specific job role and requirements.

    Job Description: {job_description}
    """
    return generate_response(prompt, "You are an expert in creating relevant interview questions based on job descriptions.", temperature, max_tokens)

with gr.Blocks(css=CSS, theme="Nymbo/Nymbo_Theme") as demo:
    gr.HTML(TITLE)

    with gr.Tab("Resume Analyzer"):
        gr.HTML(RESUME_ANALYZER_INSTRUCTIONS)
        with gr.Row():
            with gr.Column():
                with_job_description = gr.Checkbox(
                    label="Analyze with Job Description",
                    value=True,
                    info="Uncheck this box for a general resume analysis without a specific job description."
                )
                job_description = gr.Textbox(label="Job Description", lines=5)
                resume_file = gr.File(label="Upload Resume (PDF or DOCX)")
            with gr.Column():
                resume_content = gr.Textbox(label="Parsed Resume Content", lines=10)
        analyze_btn = gr.Button("Analyze Resume")
        output = gr.Markdown()

    with gr.Tab("Content Rephraser"):
        text_to_rephrase = gr.Textbox(label="Text to Rephrase", lines=5)
        rephrase_btn = gr.Button("Rephrase")
        rephrased_output = gr.Markdown()

    with gr.Tab("Cover Letter Generator"):
        gr.HTML(COVER_LETTER_INSTRUCTIONS)
        gr.HTML(COVER_LETTER_DISCLAIMER)
        generate_cl_btn = gr.Button("Generate Cover Letter")
        cover_letter_output = gr.Markdown()

    with gr.Tab("Interview Questions Generator"):
        gr.HTML(INTERVIEW_QUESTIONS_INSTRUCTIONS)
        gr.HTML(INTERVIEW_QUESTIONS_DISCLAIMER)
        interview_job_description = gr.Textbox(label="Job Description for Interview Questions", lines=5)
        generate_iq_btn = gr.Button("Generate Interview Questions")
        interview_questions_output = gr.Markdown()

    with gr.Accordion("⚙️ Parameters", open=False):
        temperature = gr.Slider(
            minimum=0, maximum=1, step=0.1, value=0.5, label="Temperature",
        )
        max_tokens = gr.Slider(
            minimum=50, maximum=1024, step=1, value=1024, label="Max tokens",
        )
        
    def update_job_description_visibility(with_job_description):
        return gr.update(visible=with_job_description)

    with_job_description.change(
        update_job_description_visibility,
        inputs=[with_job_description],
        outputs=[job_description]
    )

    def process_resume(file):
        if file is not None:
            file_type = file.name.split('.')[-1].lower()
            if file_type == 'pdf':
                return extract_text_from_pdf(file.name)
            elif file_type == 'docx':
                return extract_text_from_docx(file.name)
        return ""

    resume_file.upload(process_resume, resume_file, resume_content)

    analyze_btn.click(
        analyze_resume,
        inputs=[resume_content, job_description, with_job_description, temperature, max_tokens],
        outputs=[output]
    )

    rephrase_btn.click(
        rephrase_text,
        inputs=[text_to_rephrase, temperature, max_tokens],
        outputs=[rephrased_output]
    )

    generate_cl_btn.click(
        generate_cover_letter,
        inputs=[resume_content, job_description, temperature, max_tokens],
        outputs=[cover_letter_output]
    )

    generate_iq_btn.click(
        generate_interview_questions,
        inputs=[interview_job_description, temperature, max_tokens],
        outputs=[interview_questions_output]
    )

    gr.HTML(FOOTER_TEXT)

if __name__ == "__main__":
    demo.launch()
