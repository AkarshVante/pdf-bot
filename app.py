import streamlit as st
import os
from dotenv import load_dotenv
import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import AstraDB as AstraDBVectorStore
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.chat_models import ChatAnthropic
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline
import cassio
from openai import RateLimitError, AuthenticationError
import cassandra

# Load environment variables
load_dotenv()

# Environment variables
ASTRA_DB_TOKEN = os.getenv("ASTRA_DB_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
ASTRA_DB_SECURE_BUNDLE_PATH = os.getenv("ASTRA_DB_SECURE_BUNDLE_PATH")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

# Cached model loading functions
@st.cache_resource
def load_embedding_model(provider):
    if provider == "openai":
        return OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    elif provider == "anthropic":
        return AnthropicEmbeddings(api_key=ANTHROPIC_API_KEY)
    elif provider == "huggingface":
        return HuggingFaceEmbeddings()
    raise ValueError(f"Unknown embedding provider: {provider}")

@st.cache_resource
def load_llm(provider):
    if provider == "openai":
        return ChatOpenAI(api_key=OPENAI_API_KEY)
    elif provider == "anthropic":
        return ChatAnthropic(api_key=ANTHROPIC_API_KEY)
    elif provider == "huggingface":
        return HuggingFacePipeline.from_model_id(model_id="gpt2", task="text-generation")
    raise ValueError(f"Unknown LLM provider: {provider}")

# Streamlit UI
st.title("PDF Q&A Chatbot")
st.markdown("""
Upload a PDF file to start asking questions about its content. The app extracts text, processes it, and uses AI to provide answers.
""")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", help="Upload a text-based PDF file.")

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
            if not text.strip():
                st.error("The PDF appears to be image-only or empty. Please upload a text-based PDF.")
                st.stop()
            else:
                st.success("PDF text extracted successfully!")
        except PyPDF2.errors.PdfReadError:
            st.error("The uploaded PDF is corrupt or unreadable. Please try another file.")
            st.stop()
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            st.stop()

        # Split text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        chunks = text_splitter.split_text(text)

        # Initialize embedding model with fallbacks
        embedding_providers = [EMBEDDING_PROVIDER, "anthropic", "huggingface"]
        embedding_model = None
        for provider in embedding_providers:
            try:
                embedding_model = load_embedding_model(provider)
                embedding_model.embed_query("test")  # Test the model
                break
            except (RateLimitError, AuthenticationError) as e:
                st.warning(f"{provider.capitalize()} embedding failed ({str(e)}), trying next provider.")
        if not embedding_model:
            st.error("All embedding providers failed. Please check your API keys and network.")
            st.stop()

        # Connect to AstraDB and initialize vector store
        try:
            cassio.init(
                token=ASTRA_DB_TOKEN,
                database_id=ASTRA_DB_ID,
                secure_connect_bundle=ASTRA_DB_SECURE_BUNDLE_PATH
            )
            vector_store = AstraDBVectorStore(
                embedding=embedding_model,
                collection_name="pdf_chunks"
            )
            vector_store.add_texts(chunks)
        except ValueError as e:
            st.error(f"Invalid AstraDB credentials: {str(e)}")
            st.stop()
        except cassandra.cluster.NoHostAvailable:
            st.error("Network failure: AstraDB metadata service unreachable. Please check your connection or try again later.")
            st.stop()
        except Exception as e:
            st.error(f"Failed to initialize AstraDB: {str(e)}")
            st.stop()

    # Question input form
    with st.form(key="question_form"):
        question = st.text_input("Ask a question about the PDF:", help="Enter your question here.")
        submit_button = st.form_submit_button(label="Submit")

    if submit_button and question:
        with st.spinner("Generating answer..."):
            try:
                # Embed the question
                query_embedding = embedding_model.embed_query(question)
                # Search for relevant chunks
                docs = vector_store.similarity_search_by_vector(query_embedding, k=1)
                if not docs:
                    st.warning("No relevant information found in the PDF.")
                else:
                    context = docs[0].page_content

                    # Initialize LLM with fallbacks
                    llm_providers = [LLM_PROVIDER, "anthropic", "huggingface"]
                    llm = None
                    for provider in llm_providers:
                        try:
                            llm = load_llm(provider)
                            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
                            answer = llm.invoke(prompt).content if provider != "huggingface" else llm(prompt)
                            st.markdown("**Answer:**")
                            st.write(answer.strip())
                            with st.expander("Most Relevant Passage"):
                                st.write(context)
                            break
                        except (RateLimitError, AuthenticationError) as e:
                            st.warning(f"{provider.capitalize()} LLM failed ({str(e)}), trying next provider.")
                    if not llm:
                        st.error("All LLM providers failed. Please check your API keys and network.")
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")