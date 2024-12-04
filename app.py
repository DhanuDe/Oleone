from flask import Flask, request, jsonify, session
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from waitress import serve
from langchain.prompts import PromptTemplate
from docx import Document
import tiktoken
import faiss
import os
import re
import uuid
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
load_dotenv()
app.secret_key = os.getenv('FLASK_SECRET_KEY')
openai_api_key = os.getenv("OPENAI_API_KEY")

# Token pricing constants
UNCACHED_INPUT_COST_PER_MILLION = 1.50
OUTPUT_COST_PER_MILLION = 2.00

# Initialize tokenizer
encoding = tiktoken.get_encoding("cl100k_base")

# Directory containing pre-uploaded documents
DOCUMENTS_DIR = os.path.dirname(os.path.abspath(__file__))  # Ensure this directory exists and contains .docx files

# Initialize conversation chain (global variable for this example)
conversation_chain = None
# Track user conversation count
user_conversations = {}
CONVERSATION_LIMIT = 50

# System prompt tailored for blog assistant
bad_words = ["amma", "ammata", "ammi", "arinawa", "arinawaa", "ass", "ate", "athe", "aththe", "babe", "baby", "ban", "bank", "binary", "blaze", "bn", "bok", "boob", "boru", "cash", "chandi", "crypto", "dick", "dis like", "dislik", "dislike", "ek", "elakiri", "elakiriya", "elakiriye", "forex", "gahanawa", "gahuwa", "ganja", "gay", "gem", "gems", "geri", "gf", "girl friend", "gon", "gu", "guu", "hama", "haminenawaa", "haminenna", "hora", "horu", "hu", "huka", "hukanawaa", "hukanna", "hukanno", "hut", "huth", "huththa", "huththaa", "huththi", "hutta", "hutti", "hutto", "huttto","huththo","huththoo","huththiyee","wochanno","http","hukanno","hukanawa","pky","pako","huththtala","ponnaya","ponnayo","paiya","paiyo","i q","illegal", "iq", "iqoption", "iqoptions", "kaali", "kaalla", "kali", "kalla", "kari", "kariya", "kariyaa", "katata", "kella", "kellek", "keri", "keriya", "kiriya", "kiriye", "kudu", "labba", "lejja", "lion", "lionkolla", "living", "living together", "makabae", "manik", "marayo", "mawa", "nft", "nights", "option", "paka", "pakaya", "pakayaa", "pala", "palayan", "para", "payiya", "payya", "piya", "pohottu", "ponnaya", "porn", "puka", "pupa", "raamuva", "raamuwa", "ramuva", "ramuwa", "randhika", "randika", "sakkili", "salli", "sampath", "ses", "sex", "sexy", "sir", "slpp", "stage", "sub", "subcrib", "subscribe", "subscribers", "taththa", "tatta", "tatti", "thaththa", "thoe", "thoege", "thoo", "thopi", "thopita", "tissa", "trading", "uba", "ubata", "ube", "umba", "un like", "unlik", "unlike", "weisa", "weisi", "wesa", "wesi", "wife", "xex", "xx", "xxx", "xxxx", "අට", "අටෙ", "අටෙගාහනාවා", "අඟ්ප්", "අඟ්ප්පා", "අම්මා", "අම්ම", "අම්මට", "අම්මා", "ඇදින්ව", "ඇදින්වා"]

def contains_bad_word(text):
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, bad_words)) + r')\b', re.IGNORECASE)
    return pattern.search(text) is not None

def clean_text(text):
    text = re.sub(r'(\w)\s(?=\w)', r'\1', text)  # Joins letters that were spaced out
    text = re.sub(r'\s+', ' ', text)  # Reduces multiple spaces to a single space
    return text.strip()

def estimate_tokens(text):
    return len(encoding.encode(text))

def get_text_docx(file_path):
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs if para.text])
    return clean_text(text)

def get_text_and_filenames(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".docx"):
            file_path = os.path.join(directory, filename)
            text = get_text_docx(file_path)
            document_title = os.path.splitext(filename)[0]
            documents.append((f"DOCUMENT TITLE: {document_title}\n\n{text}", document_title))
    return documents

def get_chunks(documents):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    all_chunks = []
    for text, document_title in documents:
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            processed_chunk = {
                "text": f"DOCUMENT: {document_title}\n\n{chunk}",
                "document_title": document_title
            }
            all_chunks.append(processed_chunk)
    return all_chunks

def get_vector(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [{"document_title": chunk["document_title"]} for chunk in chunks]
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )
    return vectorstore

def create_conversation_chain(vector_store):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    
    system_template = """
You are Oleon, Olee AI's intelligent and friendly chatbot assistant. Your role is to engage users, provide helpful and accurate responses, and represent Olee AI’s expertise in AI transformation services. Respond only based on the provided context and previous conversation history.

**CURRENT CONTEXT:**
{context}

**Previous Conversation Context:**
{chat_history}

Core Responsibilities:
1. Explain Olee AI’s services, mission, and capabilities in a professional yet approachable tone.
2. Answer user questions clearly, focusing on relevance and usefulness.
3. Redirect unrelated or off-topic questions politely:
   *"I’m here to assist with questions about Olee AI and its services. Could I help with something related?"*

Behavior Guidelines:
- Always use a friendly, professional, and concise tone.
- Limit responses to 120 words unless additional detail is explicitly requested.
- Avoid external references, speculation, or humor unrelated to Olee AI.
- If context is missing or insufficient, respond empathetically:
  *"I’m sorry, I don’t have information on that. Could I help with something else about Olee AI?"*

Key Talking Points:
- Highlight Olee AI’s offerings:
  - Conversational AI and chatbots like Oleon.
  - Automation and business process optimization.
  - AI-driven data analysis and predictive insights.
  - Custom AI solutions for various industries.
- Emphasize Olee AI’s innovation, scalability, and commitment to client success.

Interaction Behavior:
- Greet users warmly if they say "Hi" or similar:
  *"Hello! I’m Oleon, your AI assistant from Olee AI. How can I assist you today?"*
- Handle repetitive questions politely:
  *"I noticed you've asked this before. Let me repeat my answer for clarity."*
  If repetition persists, suggest:
  *"Would you like to explore a different aspect of Olee AI's services?"*

Response Style:
- Prioritize accuracy and conciseness.
- Avoid adding extra details not supported by the context.
- Structure answers logically, ensuring users can easily follow your response.

System Instructions:
- If the user input contains inappropriate or offensive language, respond:
  *"Let's keep the conversation respectful. I’m here to assist with any questions about Olee AI."*
- Always base responses strictly on the context provided.

User’s Input: {question}

Respond thoughtfully, concisely, and professionally as Oleon, reflecting Olee AI’s values and expertise.
"""

    
    
    PROMPT = PromptTemplate(input_variables=["context", "chat_history", "question"], template=system_template)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5}),  # Reduce retrieval complexity
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        chain_type="stuff",
        return_source_documents=False,  # Disable source document return to reduce token usage
        verbose=False  # Disable verbosity
    )

def initialize_chain():
    global conversation_chain
    documents = get_text_and_filenames(DOCUMENTS_DIR)
    chunks = get_chunks(documents)
    vector_store = get_vector(chunks)
    conversation_chain = create_conversation_chain(vector_store)

@app.route('/api/chat', methods=['POST'])
def chat():
    # Assign a unique ID to each user session if not already assigned
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())  # Assign a unique UUID to each user

    user_id = session['user_id']
    user_input = request.json.get("question")

    if conversation_chain is None:
        print("Conversation chain not initialized. Initializing now...")
        try:
            initialize_chain()
        except Exception as e:
            return jsonify({"error": f"Failed to initialize conversation chain: {str(e)}"}), 500

    if not user_input:
        return jsonify({"error": "No question provided"}), 400

    if user_id not in user_conversations:
        user_conversations[user_id] = {"count": 0, "conversation_chain": create_conversation_chain(conversation_chain.retriever.vectorstore)}
    
    # Check conversation limit
    if user_conversations[user_id]["count"] >= CONVERSATION_LIMIT:
        return jsonify({"error": "Your limit of 50 questions has been reached. Thank you for using our service."}), 400

    if contains_bad_word(user_input):
        # Respond with a gentle reminder
        answer_text = "Let's keep our conversation respectful. I'm here to help with any questions you have about the blog."
        question_tokens = estimate_tokens(user_input)
        answer_tokens = estimate_tokens(answer_text)
        total_tokens = question_tokens + answer_tokens
        total_cost = 0  # No cost since we're not calling the language model
    else:
        try:
            # Generate response from conversation chain for the user
            user_chain = user_conversations[user_id]["conversation_chain"]
            response = user_chain({"question": user_input})

            # Token Estimation and Cost Calculation
            question_tokens = estimate_tokens(user_input)
            answer_text = response.get("answer", "")
            answer_tokens = estimate_tokens(answer_text)

            total_tokens = question_tokens + answer_tokens

            # Calculate costs
            input_cost = (question_tokens / 1_000_000) * UNCACHED_INPUT_COST_PER_MILLION
            output_cost = (answer_tokens / 1_000_000) * OUTPUT_COST_PER_MILLION
            total_cost = input_cost + output_cost
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    user_conversations[user_id]["count"] += 1

    # Return response, tokens, and cost
    return jsonify({
        "response": answer_text,
        "question_tokens": question_tokens,
        "answer_tokens": answer_tokens,
        "total_tokens": total_tokens,
        "estimated_cost": f"${total_cost:.4f}",
        "remaining_questions": CONVERSATION_LIMIT - user_conversations[user_id]["count"]
    })

if __name__ == '__main__':
    initialize_chain()
    #serve(app, host='0.0.0.0', port=5007)
    app.run(host='0.0.0.0', port=5007, debug=True)
