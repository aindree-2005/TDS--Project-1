import subprocess
from datetime import datetime
import json
import os
import sqlite3
from pathlib import Path
from helpers import (
    get_task_analysis,
    process_image,
    get_text_embeddings,
    find_similar_texts,
    make_request,
    validate_file_access,
    parse_task,
    analyze_task_constraints
)


def do_a1(email):
    subprocess.run([
        "uv", 
        "run",
        "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/datagen.py",
        email,
        "--root",
        "/data"  # Fixed path
    ], check=True)  # Added check=True for error handling

def do_a2():
    try:
        # First ensure prettier is installed at the correct version
        subprocess.run(["npm", "install", "-g", "prettier@3.4.2"], check=True)
        
        # Run prettier with explicit version and configuration
        subprocess.run([
            "npx", 
            "prettier@3.4.2",
            "--write",
            "--parser", "markdown",
            "/data/format.md"
        ], check=True)
        
        return 200
        
    except subprocess.CalledProcessError as e:
        print(f"Error running prettier: {e}")
        return 500
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 500


def do_a3():
    dates = Path("/data/dates.txt").read_text().splitlines()
    formats = ["%Y/%m/%d %H:%M:%S", "%Y-%m-%d", "%d-%b-%Y", "%b %d, %Y"]
    count = sum(any(datetime.strptime(d.strip(), f).weekday() == 2 for f in formats) 
                for d in dates if d.strip())
    Path("/data/dates-wednesdays.txt").write_text(str(count))

def do_a4():
    def contact_sort_key(contact):
        return (contact["last_name"].lower(), contact["first_name"].lower())  # Case-insensitive sort
    
    contacts = json.loads(Path("/data/contacts.json").read_text())
    sorted_contacts = sorted(contacts, key=contact_sort_key)
    Path("/data/contacts-sorted.json").write_text(json.dumps(sorted_contacts, indent=2))

def do_a5():
    log_dir = Path("/data/logs")
    log_files = sorted(log_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)[:10]
    first_lines = [log.read_text().splitlines()[0] for log in log_files]
    Path("/data/logs-recent.txt").write_text("\n".join(first_lines))

def do_a6():
    index = {}
    docs_dir = Path("/data/docs")
    
    for md_file in docs_dir.rglob("*.md"):
        relative_path = str(md_file.relative_to(docs_dir))
        content = md_file.read_text().splitlines()
        for line in content:
            if line.startswith("# "):
                index[relative_path] = line.lstrip("# ").strip()
                break
    
    Path("/data/docs/index.json").write_text(json.dumps(index, indent=2))

def do_a7():
    email_content = Path("/data/email.txt").read_text()
    messages = [
        {"role": "system", "content": "Extract only the sender's email address from the email content."},
        {"role": "user", "content": email_content}
    ]
    
    result = make_request({
        "model": "gpt-4o-mini",
        "messages": messages
    }, 'chat')
    
    if isinstance(result, int):
        raise Exception(f"Email extraction failed with status {result}")
    
    Path("/data/email-sender.txt").write_text(result.strip())

def do_a8():
    result = process_image("/data/credit-card.png")
    if isinstance(result, int):
        raise Exception(f"Image processing failed with status {result}")
    
    card_number = ''.join(c for c in result if c.isdigit())
    Path("/data/credit-card.txt").write_text(card_number)

def do_a9():
    comments = Path("/data/comments.txt").read_text().splitlines()
    similar_pair = find_similar_texts(comments)
    
    if isinstance(similar_pair, int):
        raise Exception(f"Finding similar texts failed with status {similar_pair}")
    
    idx1, idx2, _ = similar_pair
    Path("/data/comments-similar.txt").write_text(f"{comments[idx1]}\n{comments[idx2]}")

def do_a10():
    with sqlite3.connect("/data/ticket-sales.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
        total_sales = cursor.fetchone()[0]
        Path("/data/ticket-sales-gold.txt").write_text(str(total_sales))
