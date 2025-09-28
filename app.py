from flask import Flask, request, jsonify, make_response, render_template
from flask_cors import CORS
import json, requests, os, re
from decimal import Decimal
import sqlparse
import certifi
from dotenv import load_dotenv
from databricks import sql  # ✅ Databricks connector

# --- CONFIGURATION ---
load_dotenv()

api_key = os.getenv("API_KEY")
GROQ_API_KEY = api_key

# Databricks config
DATABRICKS_HOSTNAME = os.getenv("DATABRICKS_HOSTNAME")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

if not all([api_key, DATABRICKS_HOSTNAME, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN]):
    raise EnvironmentError("Missing one or more required environment variables.")

os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
app = Flask(__name__)
CORS(app)

# --- LOAD MULTIPLE DATABASE SCHEMAS ---
with open("all_databases_schema.json", "r") as f:
    raw_schemas = json.load(f)

schemas = {
    db_schema["database"]: {
        **{
            table["name"]: {
                "columns": {col["column_name"]: col["type"] for col in table["columns"]}
            } for table in db_schema.get("tables", [])
        },
        **{
            view["name"]: {
                "columns": {col["column_name"]: col["type"] for col in view["columns"]}
            } for view in db_schema.get("views", [])
        }
    } for db_schema in raw_schemas
}

# --- HELPER: Fully qualify table names ---
def qualify_table_names(sql_text, db_name):
    # Replace FROM table_name or JOIN table_name with FROM db_name.table_name
    def repl(match):
        table = match.group(2)
        if "." in table:  # already qualified
            return match.group(0)
        return f"{match.group(1)} {db_name}.{table}"

    sql_text = re.sub(r"\b(FROM|JOIN)\s+([a-zA-Z_][\w]*)", repl, sql_text, flags=re.IGNORECASE)
    return sql_text

# --- GENERATE SQL ---
def generate_sql(prompt, db_schema, db_name):
    schema_info = "\n".join(
        f"{table}: {', '.join([f'{col} ({dtype})' for col, dtype in db_schema[table]['columns'].items()])}"
        for table in db_schema
    )

    system_msg = (
        f"You are an expert SQL assistant. Use the following schema from database `{db_name}`:\n{schema_info}\n"
        f"If the user asks about tables or views, include both table_name and table_type.\n"
        f"The user always ask you questions about data which is inside table(all the records) so always generate a quary for a table"
        f"When user ask any question make sure see the question and database schema and then analyse carefully after that create SQL query for find the answer of that question"
        # f"For every user request, review the question and the database schema (tables, columns, types, keys). Determine the necessary tables, joins, filters, and aggregations, then generate a precise, optimized SQL query that returns the requested result."
        f"If any user ask you for all the table name or view name you should genarate a query like this SHOW TABLES IN campaign_performance; or SHOW VIEWS IN campaign_performance; this is just a example"
        f"Output valid SQL only. No explanations."
        # f"You are an expert SQL assistant. Use the following schema from database `{db_name}`:\n{schema_info}\n"
        # f"If the user asks about tables or views, include both table_name and table_type.\n"
        # f"The user always asks about data inside tables, so always generate a query for a table.\n"
        # f"Output valid SQL only. No explanations."
    )

    payload = {
        "model": "llama-3.3-70b-versatile",
        "temperature": 0,
        "max_tokens": 200,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt.strip()}
        ]
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    res = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)

    if res.status_code == 200:
        sql_text = res.json()["choices"][0]["message"]["content"]

        sql_text = re.sub(r"```sql\s*", "", sql_text, flags=re.IGNORECASE)
        sql_text = re.sub(r"\s*```$", "", sql_text)

        lines = sql_text.strip().splitlines()
        for i, line in enumerate(lines):
            if line.strip().lower().startswith(("select", "show", "with", "describe", "explain", "use")):
                sql_text = "\n".join(lines[i:])
                break

        sql_text = sql_text.replace("'your_database_name'", f"'{db_name}'")

        sql_text = qualify_table_names(sql_text, db_name)

        if re.search(r"(tables|views)", prompt, re.IGNORECASE) and "table_type" not in sql_text.lower():
            if "FROM information_schema.tables" in sql_text:
                sql_text = sql_text.replace("SELECT table_name", "SELECT table_name, table_type")

        try:
            return str(sqlparse.parse(sql_text.strip())[0])
        except Exception as parse_err:
            raise ValueError(f"SQL parsing error: {parse_err}")
    else:
        raise Exception(f"GROQ error: {res.status_code} - {res.text}")

# --- DATABASE QUERY (Databricks) ---
def query_databricks(sql_query, db_name=None):
    connection = sql.connect(
        server_hostname=DATABRICKS_HOSTNAME,
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN
    )
    cursor = connection.cursor()
    try:
        if db_name:
            cursor.execute(f"USE `{db_name}`")  # Set database explicitly
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return rows, columns
    finally:
        cursor.close()
        connection.close()

# --- INSIGHT GENERATION ---
def generate_insight(rows, columns, prompt):
    formatted = [
        {columns[i]: (float(cell) if isinstance(cell, Decimal) else cell) for i, cell in enumerate(row)}
        for row in rows
    ]

    payload = {
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.3,
        "max_tokens": 300,
        "messages": [
            {"role": "system", "content": "You're a data analyst. Provide concise insights."},
            {"role": "user", "content": f"User question: {prompt}"},
            {"role": "user", "content": f"Data:\n{json.dumps(formatted, indent=2)}"}
            # {"role": "system", "content": "You're a data analyst. Provide concise insights."},
            # {"role": "user", "content": f"User question: {prompt}"},
            # {"role": "user", "content": f"Data:\n{json.dumps(formatted, indent=2)}"}
        ]
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    res = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)

    if res.status_code == 200:
        return res.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"GROQ Insight error: {res.status_code} - {res.text}")

# --- ROUTES ---
@app.route("/")
def index():
    db_names = list(schemas.keys())
    return render_template("index.html", databases=db_names)

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(force=True)
        prompt = data.get("prompt")
        db_name = data.get("database")

        if not prompt or not db_name:
            return make_response(jsonify({"error": "Missing prompt or database"}), 400)

        if db_name not in schemas:
            return make_response(jsonify({"error": f"Unknown database: {db_name}"}), 400)

        db_schema = schemas[db_name]
        sql_query = generate_sql(prompt, db_schema, db_name)
        rows, columns = query_databricks(sql_query, db_name)
        insight = generate_insight(rows, columns, prompt) if rows else "No data found for this query."

        return jsonify({
            "sql": sql_query,
            "columns": columns,
            "rows": rows,
            "insight": insight
        })

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return make_response(jsonify({"error": str(e)}), 500)

@app.route("/test")
def test():
    return jsonify({"status": "Flask JSON working ✅"})

if __name__ == "__main__":
    app.run(debug=True)
