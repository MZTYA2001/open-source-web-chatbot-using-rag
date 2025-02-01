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
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatResult, ChatGeneration, LLMResult
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)

@dataclass
class ModelInfo:
    """Information about an Ollama model."""
    name: str
    size: int
    modified_at: datetime
    digest: str
    details: Dict[str, Any]

class OllamaError(Exception):
    """Base exception for Ollama API errors."""
    pass

class OllamaConnectionError(OllamaError):
    """Raised when there's a connection error with Ollama API."""
    pass

class OllamaAPIError(OllamaError):
    """Raised when Ollama API returns an error."""
    pass

class OllamaAPI:
    """API client for Ollama."""
    
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
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None)  # No timeout
            )
        return self._session
    
    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
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
        """Generate a response from the model."""
        try:
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
                                try:
                                    yield json.loads(line)
                                except json.JSONDecodeError as e:
                                    raise OllamaAPIError(f"Failed to decode response: {e}")
                    return response_generator()
                else:
                    return await response.json()
                    
        except aiohttp.ClientError as e:
            raise OllamaConnectionError(f"Failed to connect to Ollama API: {e}")
        except Exception as e:
            raise OllamaAPIError(f"Ollama API error: {e}")
    
    async def chat(self,
                  messages: List[Dict[str, str]],
                  model: str = "llama2",
                  stream: bool = False,
                  options: Optional[Dict[str, Any]] = None) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Chat with the model."""
        try:
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
                                try:
                                    yield json.loads(line)
                                except json.JSONDecodeError as e:
                                    raise OllamaAPIError(f"Failed to decode response: {e}")
                    return response_generator()
                else:
                    return await response.json()
                    
        except aiohttp.ClientError as e:
            raise OllamaConnectionError(f"Failed to connect to Ollama API: {e}")
        except Exception as e:
            raise OllamaAPIError(f"Ollama API error: {e}")
    
    def list_models(self) -> List[ModelInfo]:
        """List all available models."""
        try:
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
        except requests.RequestException as e:
            raise OllamaConnectionError(f"Failed to list models: {e}")
        except Exception as e:
            raise OllamaAPIError(f"Failed to list models: {e}")
    
    def pull_model(self, name: str) -> None:
        """Pull a model from the Ollama library."""
        try:
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
                        raise OllamaAPIError(status["error"])
        except requests.RequestException as e:
            raise OllamaConnectionError(f"Failed to pull model: {e}")
        except Exception as e:
            raise OllamaAPIError(f"Failed to pull model: {e}")

    def delete_model(self, name: str) -> None:
        """Delete a model."""
        try:
            response = requests.delete(
                self._make_url("delete"),
                json={"name": name}
            )
            response.raise_for_status()
        except requests.RequestException as e:
            raise OllamaConnectionError(f"Failed to delete model: {e}")
        except Exception as e:
            raise OllamaAPIError(f"Failed to delete model: {e}")

class ChatOllama(BaseChatModel):
    """Chat model implementation for Ollama."""
    
    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        context_window: int = 4096,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ):
        """Initialize ChatOllama."""
        super().__init__(**kwargs)
        self.model = model
        self.client = OllamaAPI(base_url)
        self.temperature = temperature
        self.context_window = context_window
        self.max_tokens = max_tokens
        
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "ollama"

    def _convert_messages_to_chat_params(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, str]]:
        """Convert messages to chat parameters."""
        return [
            {
                "role": self._convert_message_to_role(message),
                "content": message.content,
            }
            for message in messages
        ]

    def _convert_message_to_role(self, message: BaseMessage) -> str:
        """Convert a message to a role string."""
        if isinstance(message, ChatMessage):
            return message.role
        elif isinstance(message, HumanMessage):
            return "user"
        elif isinstance(message, AIMessage):
            return "assistant"
        elif isinstance(message, SystemMessage):
            return "system"
        else:
            raise ValueError(f"Got unknown message type: {message}")

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat response asynchronously."""
        chat_params = self._convert_messages_to_chat_params(messages)
        options = {
            "temperature": self.temperature,
            **kwargs
        }
        if self.max_tokens:
            options["num_predict"] = self.max_tokens
        if stop:
            options["stop"] = stop
            
        try:
            response = await self.client.chat(
                messages=chat_params,
                model=self.model,
                options=options
            )
            
            if run_manager:
                await run_manager.on_llm_new_token(
                    response["message"]["content"]
                )
            
            message = AIMessage(content=response["message"]["content"])
            return ChatResult(generations=[ChatGeneration(message=message)])
        except Exception as e:
            if run_manager:
                await run_manager.on_llm_error(e)
            raise

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[ChatResult, None]:
        """Stream chat response asynchronously."""
        chat_params = self._convert_messages_to_chat_params(messages)
        options = {
            "temperature": self.temperature,
            **kwargs
        }
        if self.max_tokens:
            options["num_predict"] = self.max_tokens
        if stop:
            options["stop"] = stop
            
        try:
            async for chunk in await self.client.chat(
                messages=chat_params,
                model=self.model,
                stream=True,
                options=options
            ):
                if chunk.get("message", {}).get("content"):
                    content = chunk["message"]["content"]
                    if run_manager:
                        await run_manager.on_llm_new_token(content)
                    message = AIMessage(content=content)
                    yield ChatResult(generations=[ChatGeneration(message=message)])
        except Exception as e:
            if run_manager:
                await run_manager.on_llm_error(e)
            raise

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.close()

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens in a text."""
        # This is a rough estimate, as Ollama doesn't provide a tokenizer
        return len(text.split())

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
    llm = ChatOllama(model='phi3') # "or any other model that you have"

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
    llm = ChatOllama(model='phi3') # "or any other model that you have"

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
        "input": user_input
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
