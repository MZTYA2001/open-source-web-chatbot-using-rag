import streamlit as st
from typing import List, Tuple
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# Set page config
st.set_page_config(page_title="Chat with Website", page_icon="🌐")
st.title("Chat with Website 🌐")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# URL input
url = st.text_input("Enter a website URL:")

# Process the URL when submitted
if url and st.session_state.vector_store is None:
    try:
        with st.spinner("Processing website content..."):
            # Load and process the website content
            loader = WebBaseLoader(url)
            data = loader.load()
            
            # Split the text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(data)
            
            # Create and store the vector store
            embeddings = OllamaEmbeddings(model="llama2")
            st.session_state.vector_store = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
            )
        st.success("Website content processed successfully!")
    except Exception as e:
        st.error(f"Error processing website: {str(e)}")

# Chat interface
if st.session_state.vector_store:
    # Initialize the chat model
    llm = ChatOllama(model="llama2", temperature=0.1)
    
    # Create the retrieval chain
    retriever = st.session_state.vector_store.as_retriever(
        search_kwargs={"k": 3}
    )
    
    # Create the prompt template
    template = """Answer the question based only on the following context:

Context: {context}

Chat History: {chat_history}
Human: {question}
Assistant: """

    prompt = ChatPromptTemplate.from_template(template)
    
    # Create the chain
    chain = (
        {"context": retriever, 
         "chat_history": lambda x: x["chat_history"],
         "question": lambda x: x["question"]} 
        | prompt 
        | llm
    )
    
    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            st.chat_message("user").write(message.content)
        else:
            st.chat_message("assistant").write(message.content)
    
    # Chat input
    if user_input := st.chat_input("Ask about the website content"):
        st.chat_message("user").write(user_input)
        
        # Format chat history for context
        chat_history = "\n".join([
            f"{'Human' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in st.session_state.chat_history
        ])
        
        # Add user message to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        
        try:
            # Get the response from the chain
            with st.spinner("Thinking..."):
                response = chain.invoke({
                    "question": user_input,
                    "chat_history": chat_history
                })
            
            # Add AI response to chat history
            ai_message = AIMessage(content=response.content)
            st.session_state.chat_history.append(ai_message)
            
            # Display AI response
            st.chat_message("assistant").write(ai_message.content)
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.vector_store = None
    st.rerun()
