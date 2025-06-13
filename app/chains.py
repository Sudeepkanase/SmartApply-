import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-70b-8192")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def extract_resume_info(self, resume_text):
        prompt_resume = PromptTemplate.from_template(
            """
            ### RESUME TEXT:
            {resume_text}

            ### INSTRUCTION:
            Extract the following information from the resume: Name, Skills, Projects, Experience.
            Only return the valid JSON containing the keys: `name`, `skills`, `projects`, `experience`.
            ###
            VALID JSON (NO PREAMBLE):
            """
        )
        chain_resume = prompt_resume | self.llm
        res = chain_resume.invoke(input={"resume_text": resume_text})
        try:
            json_parser = JsonOutputParser()
            resume_info = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse resume.")
        return resume_info

    def generate_email(self, name, skills, projects, experience, job_title, job_description):
        # Determine if the user is a student or candidate
        profession = "student" if not experience else "candidate"
        
        # Generate a concise, personalized email
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
    You are {profession}, applying for the role of {job_title}. Write a concise, personalized application email to the hiring manager. The email should include:
    - A **short and precise subject line** mentioning the role (e.g., "Application for Application Developer Role").
    - A professional body explaining how the candidate's skills, experience, and projects make them a great fit for the role.
    - Ensure the email is short, tailored to the job description, and avoids overly long subject lines.

    Focus solely on showcasing how the candidate aligns with the job requirements and ensure the email reflects a real-world job application style.

    ### EMAIL (NO PREAMBLE):
    """
)


        chain_email = prompt_email | self.llm
        res = chain_email.invoke({
            "job_description": job_description,
            "name": name,
            "skills": skills,
            "projects": projects,
            "experience": experience,
            "profession": profession,
            "job_title": job_title
        })
        return res.content
