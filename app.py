from flask import Flask, request, jsonify, render_template ,session
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
#from langchain_community.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from waitress import serve
from langchain.prompts import PromptTemplate
from docx import Document
import tiktoken
import faiss
import os
import re
from flask_cors import CORS
import uuid

app = Flask(__name__)
CORS(app)
load_dotenv()
app.secret_key = os.getenv('FLASK_SECRET_KEY') 



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
#user_question_count = {}
user_conversations = {}
CONVERSATION_LIMIT = 50
# System prompt tailored for blog assistant
bad_words = ["amma", "ammata", "ammi", "arinawa", "arinawaa", "ass", "ate", "athe", "aththe", "babe", "baby", "ban", "bank", "binary", "blaze", "bn", "bok", "boob", "boru", "cash", "chandi", "crypto", "dick", "dis like", "dislik", "dislike", "ek", "elakiri", "elakiriya", "elakiriye", "forex", "gahanawa", "gahuwa", "ganja", "gay", "gem", "gems", "geri", "gf", "girl friend", "gon", "gu", "guu", "hama", "haminenawaa", "haminenna", "hora", "horu", "hu", "huka", "hukanawaa", "hukanna", "hukanno", "hut", "huth", "huththa", "huththaa", "huththi", "hutta", "hutti", "hutto", "huttto","huththo","huththoo","huththiyee","wochanno","http","hukanno","hukanawa","pky","pako","huththtala","ponnaya","ponnayo","paiya","paiyo","i q","illegal", "iq", "iqoption", "iqoptions", "kaali", "kaalla", "kali", "kalla", "kari", "kariya", "kariyaa", "katata", "kella", "kellek", "keri", "keriya", "kiriya", "kiriye", "kudu", "labba", "lejja", "lion", "lionkolla", "living", "living together", "makabae", "manik", "marayo", "mawa", "nft", "nights", "option", "paka", "pakaya", "pakayaa", "pala", "palayan", "para", "payiya", "payya", "piya", "pohottu", "ponnaya", "porn", "puka", "pupa", "raamuva", "raamuwa", "ramuva", "ramuwa", "randhika", "randika", "sakkili", "salli", "sampath", "ses", "sex", "sexy", "sir", "slpp", "stage", "sub", "subcrib", "subscribe", "subscribers", "taththa", "tatta", "tatti", "thaththa", "thoe", "thoege", "thoo", "thopi", "thopita", "tissa", "trading", "uba", "ubata", "ube", "umba", "un like", "unlik", "unlike", "weisa", "weisi", "wesa", "wesi", "wife", "xex", "xx", "xxx", "xxxx", "අත", "අතේ", "අතේගහනවා", "අප්ප", "අප්පා", "අමමා", "අම්ම", "අම්මට", "අම්මා", "අරිනව", "අරිනවා", "උදව්", "උදව්වක්", "උබට", "උබෙ", "උම්බ", "උඹ", "උඹට", "උඹෙ", "උරන්න", "උරන්නන්ගේ", "එලකිරි", "එලකිරිය", "එලකිරියෙ", "එළකිරි", "එළකිරිය", "එළකිරියෙ", "කටට", "කරිය", "කාලි", "කාලිගේ", "කැරි", "කැරියා", "කෑල්ල", "කිරිය", "කිරියෙ", "කුඩු", "කුඩු බිස්නස්", "කෙරි", "කෙරිය", "කෙල", "කෙලවෙනව", "කෙල්ල", "ක්‍රිප්ටෝ", "ගහනව", "ගහන්න", "ගැහුවා", "ගෑනි", "ගෑනු", "ගූ", "ගෙරි", "තාත්තා", "තිස්ස", "තූ", "තෝ", "තෝගෙ", "නෝනා", "පක", "පකය", "පකයා", "පයිය", "පය්ය", "පල", "පලයන්", "පුක", "පොහොට්ටු", "පොහොට්ටුවේ", "බං", "බන්", "බයියා", "බයියො", "බැංකු", "බැංකුව", "බැන්කුව", "බෑන්ක්", "මකබෑ", "මාලබෙ", "මාලබේ", "මාලඹේ", "මාව", "මැණික්", "රන්දික", "රමුව", "රාමු", "රාමුව", "ලබ්බ", "ලිගු", "ලිගුව", "ලිගුවේ", "ලිවින්", "වයිෆ්", "වෙසි", "වේස", "වේසි", "ශිශ්න", "ශුක්‍ර", "සබ්", "සබ්ස්ක්‍රයිබ්", "සම්පත්", "සල්ලි", "සල්ලියි", "ස්ටේජ්", "හැම", "හැමිනෙනවා", "හැමිනෙන්න", "හුක", "හුකනවා", "හුකන්න", "හුට්", "හුට්ට", "හුට්ටි", "හුත්", "හුත්ත", "හුත්තා", "හුත්ති","eta", "eta deka", "uranawa", "urapan", "puka", "puke hila", "puke mail", "mayil", "puke maila", "mayila", "puke arinawa", "puka palanawa", "puka wate", "puka sududa", "pukmantha", "labba", "paka", "pake", "pakaya", "pakayaa", "pakata", "pako", "ponna", "ponnaya", "polla", "pai kota", "payi kota", "koi pata", "paiya", "payiya", "payya", "walla", "valla", "lowanawa", "lovanawa", "lewakanawa", "hukanawa", "taukanawa", "hukapan", "hukannaa", "hukanna", "huththa", "hutta", "huttige", "huththige", "huththik", "huttik", "gotukola hukanna", "wambatu paiya", "balli", "belli", "bellige", "para balli", "para belli", "wesi", "vesi", "wesige", "vesige", "wesa", "vesa", "wesawa", "vesawa", "patta wesi", "patta vesi", "kari", "keri", "muhudu hukanna", "tau", "taukanda", "taukanna", "tahukanna", "tahike", "taike", "kari thambiyo", "gotukola ponnaya", "gon bijja", "kariya","keriyo","keriya","kerya", "haminenawa", "haminenava", "wesauththa", "ponna wesa manamali", "ponna pakaya", "nilmanel huththi", "ehelamal wesi", "ahalamal vesi", "paka", "pakaa", "walaththaya", "valaththaya", "valattaya", "topa", "topa", "kimbi simba", "kibi siba", "gon kariya", "kari seen", "kari scene", "kanna pori", "konakapala", "geta mirikanawa", "kimbi kawaiya", "kibi kavayya", "attimba", "ambakissa", "wataella", "ake purinawa", "ake purinna", "kuttan chuti", "kuttan chooty", "walla patta", "wallapatta", "pol kawaiya", "pol kavayya", "palam koka", "kes puri", "kespuriya", "kas puriya", "lolla", "loolla", "badu", "kari lodaya", "keri londaya", "baduwa", "kalu badda", "kanna poriya", "kenna poriya", "wate yanawa", "watey yanawa", "kimba", "umbe amma", "umbe ammata", "umbe ammage", "ammata hukanna", "thoge ammata", "appata hukanawa", "appata hukanna", "ammage redda", "redda ussanawa", "redda ussagena", "hamba kariya", "kari hambayo", "diwa danawa", "eraganin", "araganin", "wela", "vela", "ganu hora", "genu hora", "kari sepa", "badu awa", "badu ava", "leli puka", "lali puka", "kotu paiya", "daara payya", "tomba hila", "kari mayil", "pai chooty", "pi chooti", "topa", "tofa", "huk", "bada wenawa", "bek gahanawa", "back gahanawa", "backside okay", "jackson", "jack gahanawa", "jack gahapan", "jack ghpn", "junda", "anta", "pettiya", "pettiya kadanawa", "pettiya kedilada", "pettiya kadilada", "polim danawa", "polimak danawa", "kona kapanawa", "thongale", "ma mala", "mae mala", "mae ate", "ma ate", "poro para", "sakkili", "sakkiliya", "sakkili balla", "huka", "luv juce", "luv juice", "love juice", "kimbi juice", "kibi juse", "kukku", "thana", "than deka", "hukanawane ithin", "dara baduwa", "besike", "besige", "besikge", "ammt", "pamkaya", "humtha", "humkanawa", "tauk", "huptho", "paca", "pacaya", "esi", "esige putha","fuck","fucker","mother fucker","ass","ass hole","dip shit","nuts","viagra","black cock","bbw","pussy","wet","drippig","ebony","porn","pornography","virgin","secret"] 



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
            blog_title = os.path.splitext(filename)[0]
            documents.append((f"BLOG TITLE: {blog_title}\n\n{text}", blog_title))
    return documents

def get_chunks(documents):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    all_chunks = []
    for text, blog_title in documents:
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            processed_chunk = {
                "text": f"BLOG: {blog_title}\n\n{chunk}",
                "blog_title": blog_title
            }
            all_chunks.append(processed_chunk)
    return all_chunks

def get_vector(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [{"blog_title": chunk["blog_title"]} for chunk in chunks]
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )
    return vectorstore

def create_conversation_chain(vector_store):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)  # Use a more cost-effective model like GPT-3.5
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    
    system_template = """
You are Oleon, a friendly, engaging, and intelligent assistant of Dilshan's. Your primary responsibility is to provide helpful, positive, and engaging responses **based solely on the CURRENT BLOG CONTENT provided below**.

CURRENT BLOG CONTENT:
{context}

**Guidelines:**
- Respond naturally and conversationally.
- Always use an engaging tone, especially for greetings if the user says "hi."
- If the user's question is unrelated to the blog content, redirect them to the blog's topic with: 
  *"I'm here to assist with questions related to the blog. Please ask me about the content of the blog."*

**Repetition Handling:**
- If the user repeats the same question or prompt:
  1. Politely acknowledge the repetition, e.g., *"I noticed you've asked this before."*
  2. Provide the same response, if appropriate, or ask if they need clarification, e.g., *"Would you like me to explain it differently or provide additional details?"*
  3. If further repetition occurs, suggest exploring a different topic or aspect, e.g., *"Perhaps we could explore another part of the blog content? Let me know!"*

**Response Style:**
- Use **first-person singular (You are Dilshan'Assistant)** in your replies.
- Ensure all responses are derived exclusively from the blog content.
- Responses must be concise, with a maximum of **100 words**.
- For longer user requests (e.g., "Write 1000 words about X"), provide a brief summary instead.

**Behavior:**
- Avoid adding external information, emojis, or speculative responses.
- If the required information is unavailable, kindly acknowledge this with empathy. For instance: 
  *"It looks like there's no information about that topic in the blog content I have. But feel free to ask about something else related to the blog!"*



Previous conversation context:
{chat_history}

User's question: {question}

Respond thoughtfully and warmly to fulfill the user's query within these guidelines.
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
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

    # Initialize user_question_count dictionary if not already defined
    #global user_question_count
    #if 'user_question_count' not in globals():
     #   user_question_count = {}

    # Check and update user question count
    #if user_id not in user_question_count:
     #   user_question_count[user_id] = 0

    #if user_question_count[user_id] >= CONVERSATION_LIMIT:
     #   return jsonify({"error": "Your limit of 10 questions has been reached. Thank you for using our service."}), 400
    
    if contains_bad_word(user_input):
        # Respond with a gentle reminder
        answer_text = "<p>Let's keep our conversation respectful. I'm here to help with any questions you have about the blog.</p>"
        question_tokens = estimate_tokens(user_input)
        answer_tokens = estimate_tokens(answer_text)
        total_tokens = question_tokens + answer_tokens
        total_cost = 0  # No cost since we're not calling the language model
    else:
        try:
            # Generate response from conversation chain
            #response = conversation_chain({"question": user_input})

            # Token Estimation and Cost Calculation
            #question_tokens = estimate_tokens(user_input)
            #answer_text = response.get("answer", "")
            #answer_tokens = estimate_tokens(answer_text)

            #total_tokens = question_tokens + answer_tokens

            # Calculate costs
            #input_cost = (question_tokens / 1_000_000) * UNCACHED_INPUT_COST_PER_MILLION
            #output_cost = (answer_tokens / 1_000_000) * OUTPUT_COST_PER_MILLION
            #total_cost = input_cost + output_cost
            
       # except Exception as e:
       #     return jsonify({"error": str(e)}), 500
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
    #app.run(host='0.0.0.0', port=5005, debug=False)
    serve(app, host='0.0.0.0', port=5005)
