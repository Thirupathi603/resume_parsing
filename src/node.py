from pathlib import Path
import re,json
from requests import get
import streamlit as st
import tempfile
import pandas as pd

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
from db import get_connection
conn = get_connection()
print("DB Connected")
def save_candidate(result, resume_name, resume_path):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO candidates
        (
            candidate_name,
            email,
            phone,
            resume_file_name,
            resume_file_path,
            ats_score,
            match_percentage,
            skills,
            education,
            experience,
            projects,
            matched_skills,
            missing_skills,
            strengths,
            weaknesses,
            hiring_recommendation
        )
        VALUES
        (
            %s,%s,%s,%s,%s,%s,%s,
            %s,%s,%s,%s,%s,%s,%s,%s,%s
        )
        """,
        (
            result["candidate_name"],
            result["email"],
            result["phone"],
            resume_name,
            resume_path,
            result["ats_score"],
            result["match_percentage"],

            json.dumps(result["skills"]),
            json.dumps(result["education"]),
            json.dumps(result["experience"]),
            json.dumps(result["projects"]),

            json.dumps(result["matched_skills"]),
            json.dumps(result["missing_skills"]),
            json.dumps(result["strengths"]),
            json.dumps(result["weaknesses"]),

            result["hiring_recommendation"]
        )
    )

    conn.commit()
    cur.close()
    conn.close()

llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)

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
    uploaded_at: list[str]

def load_resume(state:ResumeState):

    loader = PyPDFLoader(state["file_path"])

    docs = loader.load()

    raw_text = "\n".join(
        doc.page_content for doc in docs
    )

    return {
        "raw_text": raw_text
    }



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
graph_builder.add_node("extract_resume",extract_metadata)
graph_builder.add_node("validate_resume",validate_resume)
graph_builder.add_node("ats_score",ats_scoring)

graph_builder.add_edge(START, "load_resume")
graph_builder.add_edge("load_resume","extract_resume")
graph_builder.add_edge("extract_resume", "validate_resume") 
graph_builder.add_edge("validate_resume", "ats_score")
graph_builder.add_edge("ats_score",END)

graph = graph_builder.compile()
st.title("HR Resume Intelligence Platform")

page = st.sidebar.selectbox(
    "Navigation",
    [
        "ATS Analysis",
        "Dashboard"
    ]
)

# =====================================================
# DASHBOARD PAGE
# =====================================================

if page == "Dashboard":

    st.header("Candidate Dashboard")

    try:
        conn = get_connection()

        df = pd.read_sql(
            """
            SELECT *
            FROM candidates
            ORDER BY ats_score DESC
            """,
            conn
        )

        conn.close()

        st.success(f"Rows fetched: {len(df)}")

        st.write("Columns:")
        st.write(df.columns.tolist())

        if df.empty:
            st.warning("No candidates found in database.")
            st.stop()

        search = st.text_input("Search Candidate")
        filtered_df = (df.sort_values("id", ascending=False).drop_duplicates(subset=["email"], keep="first"))

        if search:
            filtered_df = df[
                df["candidate_name"].str.contains(
                    search,
                    case=False,
                    na=False
                )
            ]
        else:
            filtered_df = df

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Candidates",
            len(filtered_df)
        )

        col2.metric(
            "Average ATS",
            round(filtered_df["ats_score"].mean(), 2)
        )

        col3.metric(
            "Top ATS",
            filtered_df["ats_score"].max()
        )

        st.subheader("Candidate Ranking")

        st.dataframe(
            filtered_df,
            use_container_width=True
        )

        # Excel Download
        excel_file = "resume_ranking.xlsx"

        filtered_df.to_excel(
            excel_file,
            index=False
        )

        with open(excel_file, "rb") as f:
            st.download_button(
                label="Download Ranking",
                data=f,
                file_name="resume_ranking.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"Database Error: {e}")

    st.stop()


# =====================================================
# ATS ANALYSIS PAGE
# =====================================================

st.header("ATS Resume Analysis")

job_description = st.text_area(
    "Paste Job Description",
    value=job_description,
    height=300
)

uploaded_files = st.file_uploader(
    "Upload Resume(s)",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    all_results = []

    progress = st.progress(0)

    for idx, uploaded_file in enumerate(uploaded_files):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        try:

            result = graph.invoke(
                {
                    "file_path": temp_path,
                    "job_description": job_description
                }
            )

            try:

                save_candidate(
                    result,
                    uploaded_file.name,
                    temp_path
                )

                st.success(
                    f"Saved {result['candidate_name']} to database"
                )

            except Exception as db_error:

                st.error(
                    f"Database Save Failed: {db_error}"
                )

            all_results.append(
                {
                    "Name": result["candidate_name"],
                    "Email": result["email"],
                    "Phone": result["phone"],
                    "Resume": uploaded_file.name,
                    "ATS Score": result["ats_score"],
                    "Match %": result["match_percentage"],
                    "Matched Skills": ", ".join(
                        result["matched_skills"]
                    ),
                    "Missing Skills": ", ".join(
                        result["missing_skills"]
                    ),
                    "Recommendation": result["hiring_recommendation"]
                }
            )

        except Exception as e:

            st.error(
                f"Failed processing {uploaded_file.name}: {e}"
            )

        progress.progress(
            (idx + 1) / len(uploaded_files)
        )

    if all_results:

        ranking_df = pd.DataFrame(all_results)

        ranking_df = ranking_df.sort_values(
            by="ATS Score",
            ascending=False
        )

        search_name = st.text_input(
            "Search Candidate Ranking"
        )

        if search_name:

            ranking_df = ranking_df[
                ranking_df["Name"].str.contains(
                    search_name,
                    case=False,
                    na=False
                )
            ]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Resumes",
            len(ranking_df)
        )

        col2.metric(
            "Average ATS",
            round(
                ranking_df["ATS Score"].mean(),
                2
            )
        )

        col3.metric(
            "Highest ATS",
            ranking_df["ATS Score"].max()
        )

        st.subheader("Resume Ranking")

        st.dataframe(
            ranking_df,
            use_container_width=True
        )

        ranking_df.to_excel(
            "resume_ranking.xlsx",
            index=False
        )

        with open(
            "resume_ranking.xlsx",
            "rb"
        ) as f:

            st.download_button(
                "Download Ranking",
                data=f,
                file_name="resume_ranking.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
