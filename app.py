from fastapi import FastAPI, Query, HTTPException
from pathlib import Path
import subprocess
import json
import sqlite3
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import logging
import requests
from typing import Dict, Union, Any, List, Optional
import base64
import re
import os
import numpy as np
import urllib.request
import tempfile

app = FastAPI()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TaskAIProxy")
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AIPROXY_TOKEN}"
}

API_URLS = {
    'chat': 'https://aiproxy.sanand.workers.dev/openai/v1/chat/completions',
    'embeddings': 'https://aiproxy.sanand.workers.dev/openai/v1/embeddings',
}

TASK_FUNCTIONS = {
    "generate data": "do_a1",
    "format markdown": "do_a2",
    "count wednesdays": "do_a3",
    "sort contacts": "do_a4",
    "get recent logs": "do_a5",
    "create markdown index": "do_a6",
    "extract email sender": "do_a7",
    "extract credit card number": "do_a8",
    "find similar comments": "do_a9",
    "calculate gold ticket sales": "do_a10"
}

# Configuration: Root directory for data files
DATA_ROOT = Path(os.getenv("DATA_ROOT", "/data"))  # Default to /data if not set

def make_request(payload: Dict[str, Any], endpoint_type: str) -> Union[str, Dict, int]:
    """Make a request to the AI Proxy service"""
    try:
        response = requests.post(API_URLS[endpoint_type], headers=HEADERS, json=payload)
        if response.status_code == 200:
            result = response.json()
            return result if endpoint_type == 'embeddings' else result["choices"][0]["message"]["content"]
        return response.status_code
    except Exception as e:
        logger.exception("Request error")
        return 500

def get_text_embeddings(texts: Union[str, List[str]]) -> Union[Dict, int]:
    """Get embeddings for text(s)"""
    if isinstance(texts, str):
        texts = [texts]
    return make_request({
        "model": "text-embedding-3-small",
        "input": texts
    }, 'embeddings')

async def determine_task(task_description: str) -> str:
    """Map task description to function name using embeddings"""
    try:
        task_embedding_response = get_text_embeddings(task_description)
        if isinstance(task_embedding_response, int):
            return None

        task_embedding = task_embedding_response['data'][0]['embedding']
        task_names = list(TASK_FUNCTIONS.keys())
        function_embeddings_response = get_text_embeddings(task_names)
        
        if isinstance(function_embeddings_response, int):
            return None

        function_embeddings = [data['embedding'] for data in function_embeddings_response['data']]
        similarities = cosine_similarity([task_embedding], function_embeddings)[0]
        most_similar_index = np.argmax(similarities)
        
        return TASK_FUNCTIONS[task_names[most_similar_index]]
    except Exception:
        logger.exception("Task determination error")
        return None

def do_a1(email: str):
    """Generate data with modified datagen.py"""
    subprocess.Popen(
        [
            "uv",
            "run",
            "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py",
            f"{email}",
        ]
    )
    print("data generated successfully")

def do_a2():
    """Format markdown using prettier"""
    format_md_path = DATA_ROOT / "format.md"
    subprocess.Popen(["prettier", str(format_md_path), "--write", "--parser", "markdown"])
    print("data formatted successfully")

def do_a3():
    """Count Wednesdays from dates in a file"""
    count = 0
    date_formats = [
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d",
        "%d-%b-%Y",
        "%b %d, %Y",
    ]
    dates_file_path = DATA_ROOT / "dates.txt"
    wednesdays_file_path = DATA_ROOT / "dates-wednesdays.txt"

    with open(dates_file_path, "r") as f:
        for i in f:
            date = i.strip()
            if date:
                for format in date_formats:
                    try:
                        date_obj = datetime.strptime(date, format)
                        if date_obj.weekday() == 2:
                            count += 1
                    except ValueError:
                        continue

    with open(wednesdays_file_path, "w") as f:
        f.write(str(count))

def do_a4():
    """Sort contacts by name"""
    contacts_file_path = DATA_ROOT / "contacts.json"
    contacts_sorted_file_path = DATA_ROOT / "contacts-sorted.json"

    with open(contacts_file_path, "r") as f:
        contacts = json.load(f)
        sorted_contacts = sorted(contacts, key=lambda x: (x["last_name"], x["first_name"]))

    with open(contacts_sorted_file_path, "w") as f:
        json.dump(sorted_contacts, f)

def do_a5():
    """Get recent log files"""
    try:
        log_dir = DATA_ROOT / "logs"

        if not log_dir.exists() or not log_dir.is_dir():
            raise Exception("Logs directory not found!")

        log_files = sorted(log_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)[:10]

        first_lines = []
        for log_file in log_files:
            with open(log_file, "r") as f:
                try:
                    first_lines.append(f.readline().strip())
                except Exception as e:
                    first_lines.append(f"Error reading {log_file.name}: {str(e)}")

        output_file_path = DATA_ROOT / "logs-recent.txt"
        with open(output_file_path, "w") as f:
            f.write("\n".join(first_lines))

    except Exception as e:
        print(f"An error occurred: {e}")

def do_a6():
    """Create markdown index"""
    index = {}
    docs_dir = DATA_ROOT / "docs"
    index_file_path = DATA_ROOT / "docs/index.json"

    for md_file in docs_dir.rglob("*.md"):
        relative_path = str(md_file.relative_to(docs_dir))
        content = md_file.read_text().splitlines()
        for line in content:
            if line.startswith("# "):
                index[relative_path] = line.lstrip("# ").strip()
                break
    index_file_path.write_text(json.dumps(index, indent=2))
    return "do_a6 success"

def do_a7():
    """Extract email sender"""
    try:
        email_file_path = DATA_ROOT / "email.txt"
        email_sender_file_path = DATA_ROOT / "email-sender.txt"

        if not email_file_path.exists() or not email_file_path.is_file():
            raise Exception("Email file not found!")

        email_content = email_file_path.read_text(encoding='utf-8')

        messages = [
            {"role": "system", "content": "Extract only the sender's email address from the email content."},
            {"role": "user", "content": email_content}
        ]

        result = make_request({
            "model": "gpt-4o-mini",
            "messages": messages
        }, 'chat')

        if isinstance(result, int):
            raise Exception(f"LLM request failed with status code {result}")

        email_sender_file_path.write_text(result.strip(), encoding='utf-8')

    except Exception as e:
        print(f"An error occurred: {e}")

def do_a8():
    """Extract credit card number"""
    try:
        image_file_path = DATA_ROOT / "credit_card.png"
        output_file_path = DATA_ROOT / "credit-card.txt"

        if not image_file_path.exists() or not image_file_path.is_file():
            raise Exception("Credit card image not found!")

        with open(image_file_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')

        messages = [
            {"role": "system", "content": """You are a specialized credit card number extractor. 
            Focus ONLY on finding and extracting the 16-digit credit card number from the image.
            Return ONLY the 16 digits with no spaces or separators. Please try to differentiate between 3 and 5 clearly, don't accidentally swap them"""},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
                {"type": "text", "text": "Extract only the 16-digit credit card number from this image."}
            ]}
        ]

        result = make_request({
            "model": "gpt-4o-mini",
            "messages": messages
        }, 'chat')

        if isinstance(result, int):
            raise Exception(f"LLM request failed with status code {result}")

        card_number = ''.join(c for c in result if c.isdigit())
        if len(card_number) != 16:
            raise Exception(f"Extracted card number is not 16 digits long: {card_number}")

        output_file_path.write_text(card_number, encoding='utf-8')

    except Exception as e:
        print(f"An error occurred: {e}")

def do_a9():
    """Find similar comments"""
    try:
        comments_file_path = DATA_ROOT / "comments.txt"
        comments_similar_file_path = DATA_ROOT / "comments-similar.txt"

        if not comments_file_path.exists() or not comments_file_path.is_file():
            raise Exception("Comments file not found!")

        comments = comments_file_path.read_text(encoding='utf-8').splitlines()
        if len(comments) < 2:
            raise Exception("Not enough comments to find similarity!")

        embeddings_response = get_text_embeddings(comments)
        if isinstance(embeddings_response, int):
            raise Exception(f"Embedding request failed with status code {embeddings_response}")

        embeddings = [data['embedding'] for data in embeddings_response['data']]

        max_similarity = -1
        most_similar_pair = (0, 1)
        for i in range(len(comments)):
            for j in range(i + 1, len(comments)):
                similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_pair = (i, j)

        with open(comments_similar_file_path, "w", encoding='utf-8') as f:
            f.write(comments[most_similar_pair[0]] + "\n")
            f.write(comments[most_similar_pair[1]] + "\n")

    except Exception as e:
        print(f"An error occurred: {e}")

def do_a10():
    """Calculate gold ticket sales"""
    try:
        db_file_path = DATA_ROOT / "ticket-sales.db"
        output_file_path = DATA_ROOT / "ticket-sales-gold.txt"

        if not db_file_path.exists() or not db_file_path.is_file():
            raise Exception("Database file not found!")

        conn = None
        try:
            conn = sqlite3.connect(str(db_file_path))
            cursor = conn.cursor()
            cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
            total_sales = cursor.fetchone()[0]
        except sqlite3.Error as e:
            raise Exception(f"Database error: {e}")
        finally:
            if conn:
                conn.close()

        if total_sales is not None:
            output_file_path.write_text(str(total_sales), encoding='utf-8')
        else:
            output_file_path.write_text("0", encoding='utf-8')

    except Exception as e:
        print(f"An error occurred: {e}")

def extract_email_from_task(task_description: str) -> Optional[str]:
    """Extract email sender"""
    try:
        messages = [
            {"role": "system", "content": "Extract only the email address from the user query."},
            {"role": "user", "content": task_description}
        ]

        result = make_request({
            "model": "gpt-4o-mini",
            "messages": messages
        }, 'chat')

        if isinstance(result, int):
            raise Exception(f"LLM request failed with status code {result}")

        return result.strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# API Endpoints
@app.post("/run")
async def run_task(
    task: str = Query(..., description="Task description"),
    email: str = Query(None, description="Email for data generation")
):
    """Execute the specified task"""
    try:
        func_name = await determine_task(task)
        if not func_name:
            raise HTTPException(status_code=400, detail="Task not recognized")

        func = globals().get(func_name)
        if not func:
            raise HTTPException(status_code=500, detail="Task function not found")

        # If the task is generate_data extract email and pass it to do_a1
        if func_name == "do_a1":
            email = extract_email_from_task(task)
            if not email:
                raise HTTPException(status_code=400, detail="Email was not found in the task description,")
            result = func(email)
        else:
            result = func()

        return {"status": "success", "result": result}
    except Exception as e:
        logger.exception(f"Task execution error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
from fastapi.responses import PlainTextResponse
@app.get("/read", response_class=PlainTextResponse)
async def read_file(path: str):
    print("-" * 80)
    from pathlib import Path

    file_path = Path(path)

    # Check if the file exists and is a file
    if not file_path.is_file():
        logger.error(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")

    # Open the file and read its content
    try:
        with open(file_path, "r") as file:
            content = file.read()
        return content
    except Exception as e:
        logger.error(f"Exception from inside app.get('/read')")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")