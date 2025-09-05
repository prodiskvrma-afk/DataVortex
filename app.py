import os, sys   # system stuff first
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv   # added for env handling

# LangChain-related imports (yeah, still a bunch of them)
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# --------------------------------------------------
# load environment variables from .env
# --------------------------------------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("Missing GROQ_API_KEY in .env file. Exiting.")
    sys.exit(1)

# keep globals (kinda lazy but works for now)
all_docs = []
loaded_files = []   

# --------------------------------------------------
# initial doc loading
# --------------------------------------------------
for filename in os.listdir('.'):
    if filename.endswith('.txt'):
        try:
            loader_txt = TextLoader(filename)
            docs = loader_txt.load()
            all_docs.extend(docs)
            loaded_files.append(filename)
            print(f"--> loaded TXT: {filename}")
        except Exception as e:
            print(f"[ERR] could not load {filename}: {e}")

    elif filename.endswith('.pdf'):
        try:
            loader_pdf = PyPDFLoader(filename)
            docs = loader_pdf.load()
            all_docs.extend(docs)
            loaded_files.append(filename)
            print(f"--> loaded PDF: {filename}")
        except Exception as e:
            print(f"[ERR] could not load {filename}: {e}")

if not all_docs:
    print("No .txt or .pdf files found. Exiting.")
    sys.exit(1)

# split into chunks (rough params, can tweak later)
splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(all_docs)

print(f"Docs found: {len(all_docs)}")
print(f"Chunks made: {len(chunks)}")

# embeddings + FAISS
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
faiss_db = FAISS.from_documents(chunks, embedder)
faiss_db.save_local("faiss_index")

# init Groq chat model
groq_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3
)

# retrieval chain
qa_chain = RetrievalQA.from_llm(
    llm=groq_llm,
    retriever=faiss_db.as_retriever(),
    return_source_documents=True
)

# --------------------------------------------------
# helper: hybrid answer
# --------------------------------------------------
def get_answer(question):
    try:
        result = qa_chain.invoke({"query": question})
    except Exception as e:
        print(f"[oops] retrieval failed: {e}")
        return "Something went wrong.", []

    answer = result.get("result", "").strip()
    refs = result.get("source_documents", [])

    if not answer or answer.lower() in ["i don't know.", "not found", ""]:
        fallback = groq_llm.invoke(question)
        answer = getattr(fallback, "content", str(fallback))
        refs = []

    return answer, refs


# --------------------------------------------------
# Flask app
# --------------------------------------------------
app = Flask(__name__, template_folder=".")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("query") if data else None
    print(f"[incoming]: {question}")

    if not question:
        return jsonify({"error": "No query provided"}), 400

    try:
        reply, refs = get_answer(question)
    except Exception as e:
        print(f"[FAIL] get_answer blew up: {e}")
        return jsonify({"error": "Internal server error"}), 500

    return jsonify({
        "answer": reply,
        "sources": [doc.metadata.get("source", "??") for doc in refs]
    })

@app.route("/add_document", methods=["POST"])
def add_document():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded = request.files['file']
    if uploaded.filename == '':
        return jsonify({"error": "Filename missing"}), 400

    fname_safe = secure_filename(uploaded.filename)

    if not (fname_safe.endswith('.txt') or fname_safe.endswith('.pdf')):
        return jsonify({"error": "Only .txt/.pdf allowed"}), 400

    try:
        uploaded.save(fname_safe)
        loader = TextLoader(fname_safe) if fname_safe.endswith('.txt') else PyPDFLoader(fname_safe)
        new_docs = loader.load()

        if not new_docs:
            return jsonify({"error": "File had no readable content"}), 400

        new_chunks = splitter.split_documents(new_docs)
        faiss_db.add_documents(new_chunks)
        faiss_db.save_local("faiss_index")

        return jsonify({"message": f"Added {fname_safe} ({len(new_chunks)} chunks)"}), 200

    except Exception as e:
        return jsonify({"error": f"Upload failed: {e}"}), 500

@app.route("/shutdown", methods=["POST"])
def shutdown():
    os._exit(0)
    return jsonify({"message": "Server shutting down..."})

if __name__ == "__main__":
    app.run(debug=True)  # keep debug=True for dev
