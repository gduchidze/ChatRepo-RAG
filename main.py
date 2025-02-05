from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import re
import base64
import logging
from typing import Optional, Tuple, List, Dict
from dotenv import load_dotenv
from openai import OpenAI
import requests

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)


class GitHubRAG:
    def __init__(self, openai_api_key: str, github_token: str, model: str = 'gpt-4o-mini'):
        self.openai_api_key = openai_api_key
        self.github_token = github_token
        self.chat_model = model
        self.client = OpenAI(api_key=self.openai_api_key)
        self.embedding_model = "text-embedding-3-small"
        self.headers = {"Authorization": f"token {self.github_token}"}

    @staticmethod
    def parse_github_url(url: str) -> Tuple[Optional[str], Optional[str]]:
        pattern = r"https://github\.com/([^/]+)/([^/]+)"
        match = re.match(pattern, url)
        return match.groups() if match else (None, None)

    def fetch_repository_files(self, owner: str, repo: str) -> List[Dict[str, str]]:
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
        response = requests.get(api_url, headers=self.headers)

        if response.status_code == 403:
            logging.error("GitHub API rate limit exceeded.")
            raise Exception("GitHub API rate limit exceeded.")

        response.raise_for_status()
        return response.json()

    def fetch_file_content(self, file_url: str) -> str:
        response = requests.get(file_url, headers=self.headers)

        if response.status_code != 200:
            logging.error(f"Error fetching file: {file_url}")
            return ""

        content = response.json().get("content", "")
        return base64.b64decode(content).decode("utf-8") if content else ""

    def read_code_files(self, owner: str, repo: str) -> List[Dict[str, str]]:
        documents = []
        allowed_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.md', '.txt'}

        try:
            files = self.fetch_repository_files(owner, repo)
        except Exception as e:
            logging.error(f"Failed to fetch files: {e}")
            return []

        for file in files:
            if any(file['name'].endswith(ext) for ext in allowed_extensions):
                try:
                    content = self.fetch_file_content(file['url'])
                    documents.append({'content': content, 'path': file['path']})
                except Exception as e:
                    logging.warning(f"Failed to read file {file['name']}: {e}")

        return documents

    def generate_response(self, query: str, documents: List[Dict[str, str]]) -> str:
        if not documents:
            return "No relevant documents found in the repository."

        context = "\n\n".join([f"File: {doc['path']}\n{doc['content']}" for doc in documents[:3]])
        messages = [
            {"role": "system",
             "content": "You are a helpful assistant that answers questions about code repositories."},
            {"role": "user",
             "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer the question based on the context provided. If you can't find relevant information in the context, say so."}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            return f"Error generating response: {e}"

    def query(self, github_url: str, question: str) -> str:
        owner, repo = self.parse_github_url(github_url)
        if not owner or not repo:
            return 'Invalid GitHub repo URL'

        documents = self.read_code_files(owner, repo)
        return self.generate_response(question, documents)


load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
github_token = os.getenv('GITHUB_TOKEN')

if not openai_api_key or not github_token:
    raise ValueError("OPENAI_API_KEY or GITHUB_TOKEN not found in environment variables")

rag_system = GitHubRAG(openai_api_key, github_token)


@app.route('/query', methods=['POST'])
def process_query():
    try:
        data = request.get_json(force=True, silent=True)
        logging.info(f"Received Data: {data}")

        if not data or 'github_url' not in data or 'question' not in data:
            return jsonify({'error': 'Missing required fields: github_url and question'}), 400

        response = rag_system.query(data['github_url'], data['question'])
        return jsonify({'response': response})

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)