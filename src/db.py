import psycopg2

def get_connection():
    return psycopg2.connect(
        host="hrsume.postgres.database.azure.com",
        database="postgres",   # change later to hr_resume_db
        user="thirupathi",
        password="Vaishu12@#",
        port=5432,
        sslmode="require"
    )
print(f'''select * from candidates;''')
if __name__ == "__main__":
    conn = get_connection()
    print("Connected Successfully")
    conn.close()
