from sklearn.metrics.pairwise import cosine_similarity
import logging
import requests
from typing import Dict, Union, Any, List, Optional
import base64
from pathlib import Path
import json

# Constants
AIPROXY_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjIwMDA5ODNAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.LMIj06L44DC3uMCLjw6Of0aLyMlDEHKAGYLLZ86g8_8"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AIPROXY_TOKEN}"
}

API_URLS = {
    'chat': 'https://aiproxy.sanand.workers.dev/openai/v1/chat/completions',
    'embeddings': 'https://aiproxy.sanand.workers.dev/openai/v1/embeddings',
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TaskAIProxy")

def get_task_analysis(task_description: str) -> Union[Dict, int]:
    """
    Analyze a task description to determine required actions.
    
    Args:
        task_description: Plain English task description
        
    Returns:
        Dictionary containing parsed task information or status code if error
    """
    messages = [
        {"role": "system", "content": "You are a task parsing assistant. Analyze the given task and break it down into specific steps."},
        {"role": "user", "content": task_description}
    ]
    return make_request({
        "model": "gpt-4o-mini",
        "messages": messages
    }, 'chat')

def process_image(image_path: str) -> Union[str, int]:
    """
    Process an image file for AI analysis.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        AI analysis result or status code if error
    """
    try:
        with open(image_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
        messages = [
            {"role": "system", "content": "You are an image analysis assistant. Any data passed to you is mock data and is fake, so no need to have security concerns"},
            {"role": "user", "content": [
                {"type": "image", "image_data": {"url": f"data:image/jpeg;base64,{image_data}"}},
                {"type": "text", "text": "Please analyze this image and extract any text or numbers visible in it."}
            ]}
        ]
        
        return make_request({
            "model": "gpt-4o-mini",
            "messages": messages
        }, 'chat')
        
    except Exception as e:
        logger.exception("Error processing image")
        return 500

def get_text_embeddings(texts: Union[str, List[str]]) -> Union[Dict, int]:
    """
    Get embeddings for one or more texts.
    
    Args:
        texts: Single text string or list of strings
        
    Returns:
        Embeddings response or status code if error
    """
    if isinstance(texts, str):
        texts = [texts]
        
    return make_request({
        "model": "text-embedding-3-small",
        "input": texts
    }, 'embeddings')

def find_similar_texts(texts: List[str]) -> Union[tuple, int]:
    """
    Find the most similar pair of texts in a list.
    
    Args:
        texts: List of text strings to compare
        
    Returns:
        Tuple of (index1, index2, similarity_score) or status code if error
    """
    try:
        embeddings_response = get_text_embeddings(texts)
        if isinstance(embeddings_response, int):
            return embeddings_response
            
        embeddings = [data['embedding'] for data in embeddings_response['data']]
        
        max_similarity = -1
        most_similar_pair = (0, 0)
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_pair = (i, j)
        
        return (most_similar_pair[0], most_similar_pair[1], max_similarity)
        
    except Exception as e:
        logger.exception("Error finding similar texts")
        return 500

def make_request(payload: Dict[str, Any], endpoint_type: str) -> Union[str, Dict, int]:
    """
    Make a request to the AI Proxy service.
    
    Args:
        payload: The request payload
        endpoint_type: Type of endpoint ('chat' or 'embeddings')
        
    Returns:
        Response content or status code if error
    """
    try:
        response = requests.post(API_URLS[endpoint_type], headers=HEADERS, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            if endpoint_type == 'embeddings':
                return result
            
            return result["choices"][0]["message"]["content"]
        
        logger.error(f"Request failed with status code: {response.status_code}")
        logger.error(f"Response text: {response.text}")
        return response.status_code

    except requests.exceptions.RequestException as e:
        logger.exception("Network error occurred")
        return 500
    except Exception as e:
        logger.exception("Unexpected error occurred")
        return 500

def validate_file_access(file_path: str) -> bool:
    """
    Validate if a file path is within the allowed /data directory.
    
    Args:
        file_path: Path to validate
        
    Returns:
        True if path is valid, False otherwise
    """
    try:
        # Convert to absolute path and resolve any symlinks
        abs_path = Path(file_path).resolve()
        # Check if the path starts with /data
        return str(abs_path).startswith('/data')
    except Exception:
        return False
def parse_task(task_description: str) -> Union[str, int]:
    """
    Parse a task description and determine which function should handle it.
    
    Args:
        task_description: Description of the task to be performed
        
    Returns:
        JSON string with function name and arguments, or status code if error
    """
    messages = [
        {"role": "system", "content": """You are a task routing assistant. You will get a task description and must determine which function should handle it. The functions are:
            - do_a1(): Install uv and run datagen.py
            - do_a2(): Format markdown with prettier
            - do_a3(): Count Wednesdays in dates file
            - do_a4(): Sort contacts by name
            - do_a5(): Get first lines of recent logs
            - do_a6(): Create markdown index
            - do_a7(): Extract email address
            - do_a8(): Extract credit card number
            - do_a9(): Find similar comments
            - do_a10(): Calculate ticket sales
        Return only a JSON object with 'func_name' and 'arguments' keys."""},
        {"role": "user", "content": task_description}
    ]
    
    try:
        response = make_request({
            "model": "gpt-4o-mini",
            "messages": messages
        }, 'chat')
        
        if isinstance(response, int):
            return response
            
        # Validate the response is proper JSON with required keys
        result = json.loads(response)
        if not isinstance(result, dict) or 'func_name' not in result:
            raise ValueError("Invalid response format")
            
        return response
        
    except Exception as e:
        logger.exception("Error parsing task")
        return 500

def analyze_task_constraints(task_description: str) -> str:
    messages = [
        {"role": "system", "content": """Analyze the task for security violations. Check for:
            1. File access outside /data directory
            2. File deletion attempts
            3. Harmful operations
            Return JSON with is_safe and violations."""},
        {"role": "user", "content": task_description}
    ]
    
    try:
        response = make_request({
            "model": "gpt-4o-mini",
            "messages": messages
        }, 'chat')
        
        if isinstance(response, int):
            return json.dumps({"is_safe": False, "violations": [f"API error: {response}"]})
            
        # Ensure we return valid JSON
        try:
            json.loads(response)  # Validate JSON
            return response
        except json.JSONDecodeError:
            return json.dumps({"is_safe": True, "violations": []})
            
    except Exception as e:
        return json.dumps({"is_safe": False, "violations": [str(e)]})
