CREATE TABLE resumes (
    id SERIAL PRIMARY KEY,
    candidate_name TEXT,
    email TEXT,
    ats_score INT,
    match_percentage INT
);