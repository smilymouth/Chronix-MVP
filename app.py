@app.route("/ask", methods=["POST"])
def ask_ai():
    data = request.get_json()
    question = data.get("query", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Simulate a short, fast response
    return jsonify({
        "response": f"AI response for: '{question}'"
    })
