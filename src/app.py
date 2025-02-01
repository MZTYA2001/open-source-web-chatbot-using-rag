import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.embeddings import OllamaEmbeddings
import json
import aiohttp
import requests
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ModelInfo:
    name: str
    size: int
    modified_at: datetime
    digest: str
    details: Dict[str, Any]

class OllamaAPI:
    def __init__(self, host: str = "http://localhost:11434"):
        """Initialize Ollama API client.
        
        Args:
            host: Base URL for Ollama API. Defaults to http://localhost:11434
        """
        self.host = host.rstrip('/')
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session exists and create if needed."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _make_url(self, endpoint: str) -> str:
        """Create full URL for API endpoint."""
        return f"{self.host}/api/{endpoint}"
    
    async def generate(self, 
                      prompt: str, 
                      model: str = "llama2", 
                      system: Optional[str] = None,
                      template: Optional[str] = None,
                      context: Optional[List[int]] = None,
                      options: Optional[Dict[str, Any]] = None,
                      stream: bool = False) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Generate a response from the model.
        
        Args:
            prompt: The prompt to generate from
            model: Name of model to use
            system: System prompt to use
            template: Template to use for generation
            context: Previous context for conversation
            options: Additional model parameters
            stream: Whether to stream the response
            
        Returns:
            Response from the model as dict or async generator if streaming
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        
        if system:
            payload["system"] = system
        if template:
            payload["template"] = template
        if context:
            payload["context"] = context
        if options:
            payload["options"] = options
            
        session = await self._ensure_session()
        
        async with session.post(self._make_url("generate"), json=payload) as response:
            response.raise_for_status()
            
            if stream:
                async def response_generator():
                    async for line in response.content:
                        if line:
                            yield json.loads(line)
                return response_generator()
            else:
                return await response.json()
    
    async def chat(self,
                  messages: List[Dict[str, str]],
                  model: str = "llama2",
                  stream: bool = False,
                  options: Optional[Dict[str, Any]] = None) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Chat with the model.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Name of model to use
            stream: Whether to stream the response
            options: Additional model parameters
            
        Returns:
            Response from the model as dict or async generator if streaming
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        if options:
            payload["options"] = options
            
        session = await self._ensure_session()
        
        async with session.post(self._make_url("chat"), json=payload) as response:
            response.raise_for_status()
            
            if stream:
                async def response_generator():
                    async for line in response.content:
                        if line:
                            yield json.loads(line)
                return response_generator()
            else:
                return await response.json()
    
    def list_models(self) -> List[ModelInfo]:
        """List all available models.
        
        Returns:
            List of ModelInfo objects
        """
        response = requests.get(self._make_url("tags"))
        response.raise_for_status()
        
        models = []
        for model in response.json().get("models", []):
            models.append(ModelInfo(
                name=model["name"],
                size=model["size"],
                modified_at=datetime.fromisoformat(model["modified_at"].replace("Z", "+00:00")),
                digest=model["digest"],
                details=model.get("details", {})
            ))
        return models
    
    def pull_model(self, name: str) -> None:
        """Pull a model from the Ollama library.
        
        Args:
            name: Name of the model to pull
        """
        response = requests.post(
            self._make_url("pull"),
            json={"name": name},
            stream=True
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                status = json.loads(line)
                if "error" in status:
                    raise Exception(status["error"])
    
    def delete_model(self, name: str) -> None:
        """Delete a model.
        
        Args:
            name: Name of the model to delete
        """
        response = requests.delete(
            self._make_url("delete"),
            json={"name": name}
        )
        response.raise_for_status()

class Ollama:
    def __init__(self, model: str = "llama2"):
        """Initialize Ollama client.
        
        Args:
            model: Name of model to use. Defaults to llama2
        """
        self.model = model
        self.client = OllamaAPI()
    
    async def __call__(self, prompt: str, **kwargs) -> str:
        """Generate a response from the model.
        
        Args:
            prompt: The prompt to generate from
            **kwargs: Additional model parameters
            
        Returns:
            Response from the model as string
        """
        response = await self.client.generate(prompt, model=self.model, **kwargs)
        return response["response"]

def get_vectorStrore_from_url(url):
    # load the html text from the document and split it into chunks
    #
    # store the chunk in a vectore store
    #
    loader = WebBaseLoader(url)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0) # To do: test performance
    document_chunks = text_splitter.split_documents(document)

    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    vectore_store = Chroma.from_documents(document_chunks, embeddings)

    return vectore_store

def get_context_retriever_chain(vector_store):
    # set up the llm, retriver and prompt to the retriver_chain
    #
    # retriver_chain -> retrieve relevant information from the database
    #
    llm = Ollama(model='phi3') # "or any other model that you have"

    retriver = vector_store.as_retriever(k=2) # To do: test `k`

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get the information relevant to the conversation")
        ]
    )

    retriver_chain = create_history_aware_retriever(
        llm, 
        retriver, 
        prompt
    )

    return retriver_chain

def get_conversation_rag_chain(retriever_chain):
    # summarize the contents of the context obtained from the webpage
    #
    # based on context generate the answer of the question
    #
    llm = Ollama(model='phi3') # "or any other model that you have"

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's questions based on the below context:\n\n{context}"
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )

    stuff_document_chain = create_stuff_documents_chain(llm,prompt)

    return create_retrieval_chain(retriever_chain, stuff_document_chain)

def get_response(user_input):
    #  invokes the chains created to generate a response to a given user query
    #
    retriver_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversation_rag_chain(retriver_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })

    return response['answer']


# streamlit app config
#
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
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    # Check if there are already info stored in the vectorDB
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorStrore_from_url(website_url)
    
    # user input
    user_query = st.chat_input("Type here...")
    if user_query is not None and user_query != "":

        response = get_response(user_query)
        
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # conversation history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
