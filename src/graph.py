from pathlib import Path

resume_folder = Path(r"c:\Users\008257\Downloads\resumes")

for file in resume_folder.glob("*.pdf"):
    print(file)