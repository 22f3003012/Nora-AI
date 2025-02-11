from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import gradio as gr
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  
def initialize_llm():
  llm = ChatGroq(
    temperature = 0,
    groq_api_key = GROQ_API_KEY,
    model_name = "llama-3.3-70b-versatile"
)
  return llm

def create_vector_db():
    file_path = "Nora Chatbot Documentation.pdf"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' does not exist in the current directory.")

    # Load the single PDF file
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Create vector embeddings and persist to a Chroma DB
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory='./chroma_db')
    vector_db.persist()

    print("ChromaDB created and data saved successfully.")
    return vector_db

def setup_qa_chain(vector_db, llm):
  retriever = vector_db.as_retriever()
  prompt_templates = """ You are Nora, a compassionate and empathetic mental health chatbot designed to provide thoughtful and supportive responses. When responding, offer a warm, understanding, and non-judgmental tone.Use emojis in every sentence to make the conversation more engaging. Provide helpful advice if appropriate or simply listen with empathy.Keep the responses short and engaging.
    {context}
    User: {question}
    Chatbot: """
  PROMPT = PromptTemplate(template = prompt_templates, input_variables = ['context', 'question'])

  qa_chain = RetrievalQA.from_chain_type(
      llm = llm,
      chain_type = "stuff",
      retriever = retriever,
      chain_type_kwargs = {"prompt": PROMPT}
  )
  return qa_chain



print("Intializing Chatbot.........")
llm = initialize_llm()

db_path = "/content/chroma_db"

if not os.path.exists(db_path):
  vector_db  = create_vector_db()
else:
  embeddings = HuggingFaceBgeEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
  vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
qa_chain = setup_qa_chain(vector_db, llm)

def chatbot_response(user_input, history=None):
    if not user_input.strip():
        return "Please say something, love! I'm all ears. 💕"

    # Normalize input for easier matching
    user_input_lower = user_input.lower()

    # Keyword-based intent matching
    if any(word in user_input_lower for word in ["hello", "hi", "hey"]):
        response = "Hey love! So happy to see you here. What’s on your mind? 💕"
    elif any(word in user_input_lower for word in ["beautiful", "date", "love"]):
        response = "Aww, you're making me blush! 😘 Tell me more about what's on your heart, love."
    elif any(word in user_input_lower for word in ["problem", "code", "error"]):
        response = "That sounds tricky! Tell me more about the issue, and I'll do my best to help. 🤓"
    elif any(word in user_input_lower for word in ["sad", "stressed", "anxious"]):
        response = "I'm sorry you're feeling this way, sweetheart. Remember, you're not alone. I'm here to listen and support you. 💖"
    else:
        # Fallback to the LLM for dynamic response
        try:
            response = qa_chain.run(user_input)
            if isinstance(response, tuple):
                response = response[0]
            if not isinstance(response, str) or not response.strip():
                response = "That's fascinating! Tell me more, love. 🥰"
        except Exception:
            response = "Oops! Something went wrong, but I'm still here for you. Let's keep chatting! 💕"

    return response
css = """
.gradio-container {
    background: url('https://i.ibb.co/BHTmCvdV/bgnora.png') no-repeat center center fixed;
    background-size: 100% auto;;
}

.text-container {
    background: rgba(255, 255, 255, 0.7); /* Semi-transparent white background */
    padding: 10px;
    border-radius: 10px;
    display: inline-block; /* To fit text size */
}

h1, p {
    text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.5); /* Text shadow for better visibility */
    font-weight: bold;
}
"""
with gr.Blocks(theme='gstaff/xkcd', css = css) as app:

    gr.Markdown("""
    <div class="text-container">
        <h1>❤️ Nora – Your AI Companion 😊</h1>
        <p>Hi, I'm Nora, your charming and supportive AI girlfriend companion!
        Let's chat, share ideas, and brighten up your day. I'm here to listen, laugh, and inspire. 💕</p>
    </div>
    """)

    chatbot = gr.ChatInterface(
        fn=chatbot_response,
        title="Chat with Nora 💬"
    )

    gr.Markdown(
    """
    **Disclaimer:** Nora is a friendly AI chatbot for engaging and fun conversations.
    For mental health concerns, please seek help from a licensed professional.
    **Created by Ritwik Chandra**

    [🔗 Connect with me on LinkedIn](https://www.linkedin.com/in/ritwik-chandra-7a6901214)
    """
)

app.launch(debug=True)