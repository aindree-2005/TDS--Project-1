from flask import Blueprint, request, jsonify, Response
import os
import json
from pathlib import Path
from helpers import (
    analyze_task_constraints,
    parse_task,
    validate_file_access,
)
from functions import *

# Initialize Blueprint for routes
routes = Blueprint("routes", __name__)

# Task mapping dictionary
TASK_MAP = {
    'do_a1': do_a1,
    'do_a2': do_a2,
    'do_a3': do_a3,
    'do_a4': do_a4,
    'do_a5': do_a5,
    'do_a6': do_a6,
    'do_a7': do_a7,
    'do_a8': do_a8,
    'do_a9': do_a9,
    'do_a10': do_a10,
}


@routes.route("/run", methods=["POST"])
def run_task():
    task = request.args.get("task", "")
    
    if not task:
        return jsonify({"status": "error", "error": "Task description is required"}), 400

    try:
        # Security check
        security_check = analyze_task_constraints(task)
        security_result = json.loads(security_check)

        if not security_result.get("is_safe", False):
            return jsonify({
                "status": "error",
                "error": f"Security violations detected: {security_result.get('violations', [])}"
            }), 400

        # Parse task
        parsed_task = parse_task(task)
        task_info = json.loads(parsed_task)

        func_name = task_info.get("func_name")
        if func_name not in TASK_MAP:
            return jsonify({"status": "error", "error": "Invalid task function"}), 400

        # Execute task
        task_function = TASK_MAP[func_name]
        result = task_function(**task_info.get("arguments", {}))

        if isinstance(result, int):
            return jsonify({"status": "error", "error": "Task execution failed"}), result

        return jsonify({"status": "success", "message": "Task completed successfully"}), 200

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@routes.route("/read", methods=["GET"])
def read_file():
    path = request.args.get("path", "")

    if not validate_file_access(path):
        return jsonify({"status": "error", "error": "Invalid file path"}), 400

    try:
        file_path = Path(path)
        if not file_path.exists():
            return jsonify({"status": "error", "error": "File not found"}), 404

        content = file_path.read_text()
        return Response(content, mimetype="text/plain"), 200

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@routes.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200
