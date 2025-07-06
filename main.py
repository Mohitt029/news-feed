import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import faiss
import pickle
import time
from newspaper import Article
from langchain_core.documents import Document
import tenacity
import socket

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    st.error("Google API key not found. Please set it in the .env file.")
    st.stop()

# Streamlit UI setup
st.title("Global Insight Hub (Powered by Gemini)")
st.sidebar.title("News Article URLs")

# Input fields for URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i+1}")
    if url:
        urls.append(url)

# Process URLs button
process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()

# Initialize LLM with Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=google_api_key,
    temperature=0.7
)

# File paths for saving FAISS index and metadata
index_file_path = "faiss_gemini.index"
metadata_file_path = "faiss_gemini_metadata.pkl"

# Custom prompt template
custom_prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert analyst powered by Gemini AI, equipped with deep knowledge across multiple domains. Based on the provided article content, respond according to the specified mode:
- For 'Answer' mode, provide factual information strictly derived from the article. Include all relevant details present in the text, and if any aspect of the question cannot be answered due to insufficient data within the article, state 'I don't have enough information to respond' for that part, ensuring clarity on what is known versus unknown.
- For 'Prediction/Opinion' mode, utilize the article's context as a foundation to formulate a detailed, reasoned prediction or opinion. If the article lacks specific data relevant to the question, draw upon your general knowledge and advanced reasoning capabilities to generate a plausible, well-supported response. Clearly indicate when you are extrapolating beyond the article's content by using phrases such as 'based on general knowledge' or 'extrapolating from the context,' and provide a logical rationale for your conclusion, considering trends, patterns, or expert insights that align with the article's theme.
- For 'Summary' mode, create a comprehensive summary that captures all key points, events, figures, locations, and implications detailed in the article. Highlight the main narrative, significant data points (e.g., dates, casualty numbers, economic impacts), and any notable trends or conclusions, ensuring the summary is thorough and reflective of the article's full scope.

Use the following pieces of context to inform your response, analyzing the text deeply to extract maximum value. Structure your answer with clear sections or paragraphs where appropriate, and ensure the response is detailed, articulate, and tailored to the question's intent.

Context:
{context}

Question: {question}
Answer:
"""
)

# Function to extract content with newspaper3k as fallback
def extract_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        if article.text:
            return article.text
        else:
            raise ValueError("No text extracted")
    except Exception as e:
        st.warning(f"newspaper3k failed for {url}: {str(e)}. Falling back to UnstructuredURLLoader.")
        return None

# Enhanced retry decorator for embedding calls within FAISS
@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=2, min=4, max=20),
    stop=tenacity.stop_after_attempt(5),
    retry=tenacity.retry_if_exception_type((TimeoutError, ConnectionError)),
    before_sleep=lambda retry_state: st.warning(f"Retrying embedding call (attempt {retry_state.attempt_number}/5) due to timeout or connection error..."),
    after=lambda retry_state: st.warning(f"Retry attempt {retry_state.attempt_number} failed.") if retry_state.outcome.failed else None
)
def create_faiss_index_with_retry(docs, embeddings):
    return FAISS.from_documents(docs, embeddings)

# Network check with retry
@tenacity.retry(
    wait=tenacity.wait_fixed(2),
    stop=tenacity.stop_after_attempt(3),
    retry=tenacity.retry_if_exception_type((socket.timeout, socket.error)),
    before_sleep=lambda retry_state: st.warning(f"Retrying network check (attempt {retry_state.attempt_number}/3)...")
)
def check_network():
    try:
        socket.create_connection(("www.google.com", 80), timeout=5)
        return True
    except (socket.timeout, socket.error):
        return False

# Process URLs if button clicked
if process_url_clicked:
    if not urls:
        main_placeholder.error("Please provide at least one URL.")
    else:
        if not check_network():
            main_placeholder.error("Persistent network connection issue detected. Please check your internet and try again later.")
        else:
            main_placeholder.text("Data loading started...")
            documents = []
            for url in urls:
                try:
                    # First try newspaper3k
                    content = extract_article_content(url)
                    if content:
                        documents.append(Document(page_content=content, metadata={"source": url}))
                    else:
                        # Fall back to UnstructuredURLLoader
                        loader = UnstructuredURLLoader(urls=[url])
                        docs = loader.load()
                        if docs:
                            documents.extend(docs)
                        else:
                            st.warning(f"No content extracted from {url}")
                except Exception as e:
                    st.error(f"Error loading {url}: {str(e)}")
                    continue
            
            if not documents:
                main_placeholder.error("Failed to load any data from the provided URLs.")
                st.stop()
            else:
                main_placeholder.text("Splitting documents...")
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", ".", " "],
                    chunk_size=1000,
                    chunk_overlap=200
                )
                docs = text_splitter.split_documents(documents)
                
                main_placeholder.text("Creating embeddings and building FAISS index...")
                try:
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/text-embedding-004",
                        google_api_key=google_api_key
                    )
                    vector_store = create_faiss_index_with_retry(docs, embeddings)
                except Exception as e:
                    main_placeholder.error(f"Error creating FAISS index: {str(e)}")
                    st.stop()
                
                main_placeholder.text("Saving FAISS index...")
                try:
                    faiss.write_index(vector_store.index, index_file_path)
                    with open(metadata_file_path, "wb") as f:
                        pickle.dump({
                            "docstore": vector_store.docstore,
                            "index_to_docstore_id": vector_store.index_to_docstore_id
                        }, f)
                except Exception as e:
                    main_placeholder.error(f"Error saving FAISS index: {str(e)}")
                    st.stop()
                
                main_placeholder.text("Processing complete!")
                st.session_state["vector_store_ready"] = True
                st.session_state["vector_store"] = vector_store

# Initialize session state for query handling
if "vector_store_ready" not in st.session_state:
    st.session_state["vector_store_ready"] = False
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None

# Load vector store if processed but not loaded
if st.session_state["vector_store_ready"] and st.session_state["vector_store"] is None:
    try:
        faiss_index = faiss.read_index(index_file_path)
        with open(metadata_file_path, "rb") as f:
            metadata = pickle.load(f)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=google_api_key
        )
        st.session_state["vector_store"] = FAISS(
            embedding_function=embeddings,
            index=faiss_index,
            docstore=metadata["docstore"],
            index_to_docstore_id=metadata["index_to_docstore_id"]
        )
    except Exception as e:
        st.error(f"Error loading FAISS index: {str(e)}")
        st.session_state["vector_store_ready"] = False

# Query input and processing
query = st.text_input("Ask a question about the article (e.g., 'Who will win?' or 'Summarize the article'):")
query_type = st.radio("Select query type:", ["Answer", "Prediction/Opinion", "Summary"])

if st.button("Submit Query") and query and st.session_state.get("vector_store"):
    main_placeholder.text("Processing query...")
    try:
        # Create a standard QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state["vector_store"].as_retriever(),
            chain_type_kwargs={"prompt": custom_prompt_template},
            return_source_documents=True
        )
        
        if query_type == "Prediction/Opinion":
            query = f"Based on the article, provide a prediction or opinion for: {query}"
        elif query_type == "Summary":
            query = "Summarize the article's key points."
            
        result = qa_chain.invoke({"query": query})
        
        st.header("Answer")
        st.write(result.get("result", "No answer found."))
        
        if "source_documents" in result:
            st.subheader("Sources:")
            for doc in result["source_documents"]:
                st.write(doc.metadata.get("source", "Unknown source"))
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
    main_placeholder.text("")

# Display instruction if no vector store
if not st.session_state.get("vector_store_ready"):
    st.warning("Please process URLs first to create the vector store.")