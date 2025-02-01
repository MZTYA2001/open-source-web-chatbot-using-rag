import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# Set page config
st.set_page_config(page_title="Chat with Website", page_icon="🌐")
st.title("Chat with Website 🌐")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# URL input
url = st.text_input("Enter a website URL:")

# Process the URL when submitted
if url and st.session_state.vector_store is None:
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

# Chat interface
if st.session_state.vector_store:
    # Initialize the chat model and chain
    llm = ChatOllama(model="llama2", temperature=0.1)
    
    # Create the retrieval chain
    retriever = st.session_state.vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question based on the following context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=None,
        combine_docs_chain_kwargs={"prompt": prompt}
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
        
        # Add user message to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        
        # Get the response from the chain
        response = chain.invoke({
            "question": user_input,
            "chat_history": [(m.type, m.content) for m in st.session_state.chat_history[:-1]]
        })
        
        # Add AI response to chat history
        ai_message = AIMessage(content=response["answer"])
        st.session_state.chat_history.append(ai_message)
        
        # Display AI response
        st.chat_message("assistant").write(ai_message.content)

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.vector_store = None
    st.rerun()
