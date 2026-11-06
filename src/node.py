from pathlib import Path
import re,json
from turtle import st
from urllib import response
import streamlit as st
import tempfile

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)
from langchain_openai import ChatOpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import StateGraph, START, END
from typing import TypedDict,List
from dotenv import load_dotenv 
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)

response=llm.invoke("hello")

print(response.content)

class ResumeState(TypedDict):
    file_path: str
    raw_text: str
    chunks: List[str]
    candidate_name: str
    email: str
    phone: str
    skills:List[str]
    is_valid: bool 
    education: List[str]
    experience: List[str] 
    projects: List[str]
    file_paths: List[str]
    job_description: str
    results: List[str]
    ats_score: int 
    match_percentage: int
    matched_skills: List[str]
    missing_skills: List[str]
    strengths: List[str]
    weaknesses: List[str]
    hiring_recommendation: str

def load_resume(state:ResumeState):

    loader = PyPDFLoader(state["file_path"])

    docs = loader.load()

    raw_text = "\n".join(
        doc.page_content for doc in docs
    )

    return {
        "raw_text": raw_text
    }



def chunk_resume(state:ResumeState):
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks=splitter.split_text(state["raw_text"]) 
    return {"chunks":chunks} 


def extract_metadata(state: ResumeState):

    prompt = ChatPromptTemplate.from_template(
        """
        Extract the following details from the resume.

        Resume:
        {resume}

        Return ONLY valid JSON.

        {{
            "candidate_name":"",
            "email":"",
            "phone":"",
            "skills":[],
            "education":[],
            "experience":[],
            "projects":[]
        }}
        """
    )

    chain = prompt | llm

    response = chain.invoke({
        "resume": state["raw_text"]
    })

    content = response.content.strip()

    print("\nLLM OUTPUT:")
    print(content)

    # Remove markdown if GPT returns ```json
    content = content.replace("```json", "")
    content = content.replace("```", "")
    content = content.strip()

    data = json.loads(content)

    return {
        "candidate_name": data.get("candidate_name", ""),
        "email": data.get("email", ""),
        "phone": data.get("phone", ""),
        "skills":data.get("skills",[]),
        "education":data.get("education",[]),
        "experience":data.get("experience",[]),
        "projects":data.get("projects",[])
    }


def validate_resume(state: ResumeState):
    is_valid= all([state["candidate_name"],state["email"],state["phone"],state["skills"]])
    return {"is_valid":is_valid}  

def ats_scoring(state: ResumeState):

    prompt = ChatPromptTemplate.from_template(
    """
    You are an enterprise ATS system.

    Compare the resume against the job description.

    Resume:
    {resume}

    Job Description:
    {job_description}

    Evaluate:

    1. Skill match (40%)
    2. Experience relevance (25%)
    3. Project relevance (15%)
    4. Education relevance (10%)
    5. Keyword coverage (10%)

    Calculate ATS Score out of 100.

    Return ONLY valid JSON.

    {{
      "ats_score": 0,
      "match_percentage": 0,
      "matched_skills": [],
      "missing_skills": [],
      "strengths": [],
      "weaknesses": [],
      "hiring_recommendation": ""
    }}
    """
    )

    chain = prompt | llm

    response = chain.invoke({
        "resume": state["raw_text"],
        "job_description": state["job_description"]
    })

    content = response.content.strip()

    content = content.replace("```json", "")
    content = content.replace("```", "")

    data = json.loads(content)

    return {
        "ats_score": data["ats_score"],
        "match_percentage": data["match_percentage"],
        "matched_skills": data["matched_skills"],
        "missing_skills": data["missing_skills"],
        "strengths": data["strengths"],
        "weaknesses": data["weaknesses"],
        "hiring_recommendation": data["hiring_recommendation"]
    }
job_description = """
Role: Data Engineer

Job Summary:
We are seeking a skilled Data Engineer with 2+ years of experience in designing, building, and maintaining scalable data pipelines and cloud-based data platforms. The ideal candidate should have strong expertise in Python, SQL, Big Data technologies, AWS/Azure cloud services, and modern data engineering tools.

Required Skills:
- Python
- SQL
- PySpark
- AWS
- Azure
- Databricks
- Dataiku
- Big Data Technologies
- ETL/ELT Development
- Data Warehousing
- Data Modeling
- Apache Spark
- Git

Responsibilities:
- Design, develop, and optimize ETL/ELT pipelines.
- Build scalable data processing solutions using PySpark and Databricks.
- Work with structured and unstructured datasets.
- Develop and maintain cloud-based data solutions on AWS and Azure.
- Implement data quality checks and monitoring processes.
- Create and optimize SQL queries, stored procedures, and data models.
- Collaborate with Data Scientists, Analysts, and Business Teams.
- Troubleshoot and resolve performance bottlenecks in data pipelines.
- Ensure data governance, security, and compliance standards are followed.

Preferred Skills:
- Docker
- Kubernetes
- CI/CD
- Airflow
- Kafka
- Delta Lake
- Snowflake
- Terraform
- GitHub Actions
- Azure Data Factory
- AWS Glue

Experience:
- 2+ Years of Experience in Data Engineering
- Experience working with cloud platforms (AWS/Azure)
- Experience in building production-grade data pipelines

Education:
- Bachelor's Degree in Computer Science, Information Technology, Data Science, Engineering, or related field

Nice to Have:
- Generative AI/Data Engineering integration experience
- MLOps exposure
- Real-time streaming data processing experience
- Certification in AWS, Azure, or Databricks
"""


graph_builder= StateGraph(ResumeState)
graph_builder.add_node("load_resume",load_resume)
graph_builder.add_node("chunk_resume", chunk_resume)
graph_builder.add_node("extract_resume",extract_metadata)
graph_builder.add_node("validate_resume",validate_resume)
graph_builder.add_node("ats_score",ats_scoring)

graph_builder.add_edge(START, "load_resume")
graph_builder.add_edge("load_resume", "chunk_resume")
graph_builder.add_edge("chunk_resume","extract_resume")
graph_builder.add_edge("extract_resume","validate_resume")
graph_builder.add_edge("validate_resume", "ats_score") 
graph_builder.add_edge("ats_score",END)

graph = graph_builder.compile()

resume_folder = Path(r"C:\Users\008257\Downloads\resumes")

all_results = []

for pdf in resume_folder.glob("*.pdf"):

    print(f"\nProcessing: {pdf.name}")

    result = graph.invoke({
        "file_path": str(pdf),
        "job_description": job_description
    })

    all_results.append({
    "name": result["candidate_name"],
    "email": result["email"],
    "phone": result["phone"],
    "resume": pdf.name,
    "ats_score": result["ats_score"],
    "match_percentage": result["match_percentage"],
    "matched_skills": ",".join(result["matched_skills"]),
    "missing_skills": ",".join(result["missing_skills"]),
    "recommendation": result["hiring_recommendation"]
})


all_results.sort(
    key=lambda x: x["ats_score"],
    reverse=True
)

print("\n===== FINAL RANKING =====")

for rank, candidate in enumerate(all_results, start=1):

    print(f"\nRank #{rank}")

    print("Candidate :", candidate["name"])
    print("Resume    :", candidate["resume"])
    print("ATS Score :", candidate["ats_score"])
    print("Match %   :", candidate["match_percentage"])
    print("Decision  :", candidate["recommendation"])


print("\n===== ATS ROUTER TEST =====")


import pandas as pd

df = pd.DataFrame(all_results)

df.to_excel(
    "resume_ranking.xlsx",
    index=False
)

print("Ranking saved successfully")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Resumes",
    len(df)
)

col2.metric(
    "Average ATS",
    round(df["ats_score"].mean(), 2)
)

col3.metric(
    "Top Score",
    df["ats_score"].max()
)

st.set_page_config(
    page_title="HR Resume Intelligence Platform",
    layout="wide"
)
search = st.text_input("Search Candidate")
df[
    df["name"]
        .str.contains(search,
                      case=False,
                      na=False)
]
st.title("HR Resume Intelligence Platform")

job_description = st.text_area(
    "Paste Job Description",
    height=300
)

uploaded_files = st.file_uploader(
    "Upload Resume(s)",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    all_results = []

    for uploaded_file in uploaded_files:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        result = graph.invoke({
            "file_path": temp_path,
            "job_description": job_description
        })

        all_results.append({
            "name": result["candidate_name"],
            "email": result["email"],
            "phone": result["phone"],
            "resume": uploaded_file.name,
            "ats_score": result["ats_score"],
            "match_percentage": result["match_percentage"],
            "matched_skills": ",".join(result["matched_skills"]),
            "missing_skills": ",".join(result["missing_skills"]),
            "recommendation": result["hiring_recommendation"]
        })

    all_results.sort(
        key=lambda x: x["ats_score"],
        reverse=True
    )

    import pandas as pd

    df = pd.DataFrame(all_results)

    st.subheader("Resume Ranking")

    st.dataframe(df)

    df.to_excel(
        "resume_ranking.xlsx",
        index=False
    )

    with open("resume_ranking.xlsx", "rb") as f:

        st.download_button(
            "Download Ranking",
            f,
            "resume_ranking.xlsx"
        )




 



