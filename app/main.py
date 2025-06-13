import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv
import PyPDF2
import re
import json

# Load environment variables
load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.4,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="meta-llama/llama-4-scout-17b-16e-instruct"
        )

    def extract_resume_info(self, resume_text):
        prompt_resume = PromptTemplate.from_template(
            """
            ### RESUME TEXT:
            {resume_text}

            ### INSTRUCTION:
            Extract the following information from the resume: Name, Skills, Projects, Experience.
            Only return the valid JSON containing the keys: `name`, `skills`, `projects`, `experience`.

            ### VALID JSON (NO PREAMBLE):
            """
        )
        res = (prompt_resume | self.llm).invoke({"resume_text": resume_text})
        try:
            return JsonOutputParser().parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse resume.")

    def generate_email(self, name, skills, projects, experience, job_title, job_description, num_emails=1):
        profession = "student" if not experience else "candidate"
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### CANDIDATE DETAILS:
            Name: {name}
            Skills: {skills}
            Projects: {projects}
            Experience: {experience}

            ### INSTRUCTION:
            You are a highly motivated {profession} applying for the role of {job_title}.
            Write {num_emails} fully formatted cold email(s) to the hiring manager. Each email MUST include:
            - A greeting (e.g., "Dear Hiring Manager")
            - A compelling opening
            - A highlight of 1–2 key skills or projects
            - A closing paragraph inviting next steps
            - A professional sign-off ("Best regards, {name}")

            **Separate each email with the exact delimiter**:
            ===EMAIL===
            """
        )
        res = (prompt_email | self.llm).invoke({
            "job_description": job_description,
            "name": name,
            "skills": skills,
            "projects": projects,
            "experience": experience,
            "profession": profession,
            "job_title": job_title,
            "num_emails": num_emails,
        })
        return res.content

    def generate_ats_score(self, job_description, resume_info):
        prompt_ats = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### RESUME INFORMATION:
            {resume_info}

            ### INSTRUCTION:
            Evaluate how well the resume matches the job description (skills, experience, fit).
            Provide a concise, point-wise ATS score (1–5):
            1. Point 1
            2. Point 2
            3. Point 3
            4. Point 4
            5. Point 5
            """
        )
        res = (prompt_ats | self.llm).invoke({
            "job_description": job_description,
            "resume_info": json.dumps(resume_info),
        })
        return res.content

# Streamlit UI
st.set_page_config(page_title="Resume Tools", layout="wide")
st.title("🚀 Smart Resume Toolkit")

chain = Chain()

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

tab1, tab2 = st.tabs(["📧 Email + ATS Score", "📄 ATS-Friendly Resume"])

with tab1:
    st.subheader("Job Email Generator & ATS Scoring")

    url_input    = st.text_input("Enter Job URL")
    resume_input = st.file_uploader("Upload Resume (PDF)", type="pdf")
    num_emails   = st.selectbox("Number of emails to generate", [1, 2, 3, 4, 5])

    if st.button("Generate Email"):
        if not url_input or not resume_input:
            st.error("Please provide both a job URL and a resume PDF.")
            st.stop()

        # Scrape job description
        resp = requests.get(url_input)
        job_desc = clean_text(BeautifulSoup(resp.text, "html.parser").get_text())

        # Extract resume info
        pdf = PyPDF2.PdfReader(resume_input)
        resume_text = "".join(p.extract_text() for p in pdf.pages)
        resume_info = chain.extract_resume_info(resume_text)

        # Generate emails + ATS score
        raw_emails = chain.generate_email(
            name=resume_info["name"],
            skills=resume_info["skills"],
            projects=resume_info["projects"],
            experience=resume_info["experience"],
            job_title="Application Developer",
            job_description=job_desc,
            num_emails=num_emails
        )
        ats_score = chain.generate_ats_score(job_desc, resume_info)

        # Display
        st.subheader("Generated Emails")
        emails = [e.strip() for e in raw_emails.split("===EMAIL===") if e.strip()]
        if num_emails == 1:
            st.text_area("Email", emails[0], height=300)
        else:
            for i, email in enumerate(emails, start=1):
                st.text_area(f"Email {i}", email, height=300, key=f"email_{i}")

        st.sidebar.header("Job Description")
        st.sidebar.text_area("", job_desc, height=200)
        st.sidebar.header("Resume Info (JSON)")
        st.sidebar.text_area("", json.dumps(resume_info, indent=2), height=200)
        st.sidebar.header("ATS Score")
        st.sidebar.text_area("", ats_score, height=200)

with tab2:
    st.subheader("Build ATS-Friendly Resume using Power Verbs ✨")
    job_url    = st.text_input("Enter Job URL for ATS Resume", key="job_url_tab2")
    resume_pdf = st.file_uploader("Upload Resume for ATS Optimization", type="pdf", key="resume_tab2")

    if st.button("Generate ATS-Friendly Resume"):
        if not job_url or not resume_pdf:
            st.error("Please provide both a job URL and a resume PDF.")
            st.stop()

        resp = requests.get(job_url)
        job_desc = clean_text(BeautifulSoup(resp.text, "html.parser").get_text())

        pdf = PyPDF2.PdfReader(resume_pdf)
        resume_text = "".join(p.extract_text() for p in pdf.pages)

        ats_prompt = PromptTemplate.from_template("""
        ### JOB DESCRIPTION:
        {job_description}

        ### ORIGINAL RESUME:
        {resume_text}

        ### INSTRUCTION:
        Optimize this resume for ATS using strong power verbs, matching job language, and keeping all original sections intact.
        Then provide a brief summary of changes.

        ### OPTIMIZED RESUME:
        """)
        resp2 = (ats_prompt | chain.llm).invoke({
            "job_description": job_desc,
            "resume_text": resume_text
        })
        optimized = resp2.content

        st.subheader("ATS-Optimized Resume")
        st.text_area("", optimized, height=400)
        st.download_button("Download ATS Resume", optimized, "ATS_Resume.txt", "text/plain")
