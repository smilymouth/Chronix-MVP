import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

@app.route("/ai", methods=["POST"])
def ai():
    data = request.json
    query = data.get("query", "")
    if not query:
        return jsonify({"response": "No query received."})

    try:
        headers = {"Content-Type": "application/json"}
        # Example Gemini-style local simulation (replace this with real API call)
        reply = f"I analyzed '{query}' and system status is nominal."

        return jsonify({"response": f"AI says: '{reply}' — running fine!"})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})

@app.route("/")
def home():
    return jsonify({"status": "Chronix AI backend running 🚀"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
