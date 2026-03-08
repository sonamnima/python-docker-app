from flask import Flask, request, jsonify
from pdf_reader import read_pdf
from vector_store import VectorStore

app = Flask(__name__)

vector_store = VectorStore()

# Load PDF at startup
pdf_text = read_pdf("sample.pdf")
vector_store.build(pdf_text)

@app.route("/")
def home():
    return "PDF AI Chatbot Running"

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data["question"]

    results = vector_store.search(question)

    answer = "\n".join(results)

    return jsonify({
        "question": question,
        "answer": answer
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)