from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

class OllamaWrapper:
    """Wrapper for Ollama integration with LangChain."""
    
    def __init__(
        self,
        model_name: str = "mistral",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        verbose: bool = True
    ):
        """Initialize Ollama wrapper with specified configuration."""
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.verbose = verbose
        
        # Initialize the LLM
        self.llm = Ollama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            verbose=verbose,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )
        
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(
            model=model_name,
            base_url=base_url
        )
    
    def create_vector_store(self, documents, persist_directory: str = None) -> Chroma:
        """Create a vector store from documents."""
        if persist_directory:
            return Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=persist_directory
            )
        return Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
    
    def load_vector_store(self, persist_directory: str) -> Chroma:
        """Load an existing vector store."""
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
    
    def create_qa_chain(
        self,
        retriever,
        prompt_template: str = None,
        memory: ConversationBufferMemory = None,
        chain_type: str = "stuff"
    ) -> RetrievalQA:
        """Create a question-answering chain."""
        if prompt_template is None:
            prompt_template = """You are a knowledgeable assistant. Use the following context to answer the question.
            
            Context: {context}
            History: {history}
            Question: {question}
            
            Answer:"""
            
        prompt = PromptTemplate(
            input_variables=["history", "context", "question"],
            template=prompt_template
        )
        
        if memory is None:
            memory = ConversationBufferMemory(
                memory_key="history",
                input_key="question",
                return_messages=True
            )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=chain_type,
            retriever=retriever,
            verbose=self.verbose,
            chain_type_kwargs={
                "verbose": self.verbose,
                "prompt": prompt,
                "memory": memory
            }
        )
    
    @staticmethod
    def split_documents(documents, chunk_size: int = 1500, chunk_overlap: int = 200):
        """Split documents into chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        return splitter.split_documents(documents)
    
    @staticmethod
    def load_webpage(url: str) -> list:
        """Load a webpage."""
        loader = WebBaseLoader(url)
        return loader.load()

def get_vector_store(url):
    documents = OllamaWrapper.load_webpage(url)
    return OllamaWrapper().create_vector_store(documents)

def get_qa_chain(vector_store):
    retriever = vector_store.as_retriever()
    return OllamaWrapper().create_qa_chain(retriever)

def get_response(user_input, qa_chain, chat_history):
    response = qa_chain({"question": user_input, "history": chat_history})
    return response

# streamlit app config
#
import streamlit as st
st.set_page_config(page_title="Lets chat with a Website", page_icon="💻")
st.title("Lets chat with a Website")

# sidebar setup
with st.sidebar:
    st.header("Setting")
    website_url = st.text_input("Type the URL here")

if website_url is None or website_url == "":
    st.info("Please enter a website URL...")

else:
    # Session State
    #
    # Check the chat history for follow the conversation
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    # Check if there are already info stored in the vectorDB
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vector_store(website_url)
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = get_qa_chain(st.session_state.vector_store)
    
    # user input
    user_query = st.chat_input("Type here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query, st.session_state.qa_chain, st.session_state.chat_history)
        st.session_state.chat_history.append(user_query)
        st.session_state.chat_history.append(response)

    # conversation history
    for message in st.session_state.chat_history:
        if isinstance(message, str):
            with st.chat_message("AI"):
                st.write(message)
        elif isinstance(message, dict):
            with st.chat_message("Human"):
                st.write(message["question"])
