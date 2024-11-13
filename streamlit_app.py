import streamlit as st
import os
import yaml
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.ollama import Ollama
from rag.data import Data  # Import the Data class from your data.py file


# Load configuration file
config_file = "config.yml"
with open(config_file, "r") as conf:
    config = yaml.safe_load(conf)


def main():
    # Set title and description of the web app
    st.title("Research Paper Downloader and Ingestor")
    st.write(
        "This application allows you to search and download papers from arXiv and ingest them into a Qdrant vector database."
    )

    # Take input from the user
    st.header("Enter your Search Query and Configuration")
    
    # Search Query Input
    query = st.text_input("Enter search query for papers (e.g., 'machine learning')", "")
    
    # Max Results Input
    max_results = st.number_input("Enter maximum number of results to download", min_value=1, max_value=100, value=5)

    # Option to download papers
    download_button = st.button("Download Papers from arXiv")

    # Option to ingest data to Qdrant
    ingest_checkbox = st.checkbox("Ingest data into Qdrant", value=False)
    
    # Action button to perform the tasks
    if download_button:
        if query:
            st.write(f"Searching and downloading papers related to: {query}")
            data = Data(config)
            data.download_papers(query, config["data_path"], max_results)
            st.success("Download Complete!")
        else:
            st.error("Please enter a search query.")

    if ingest_checkbox:
        st.write("Ingesting downloaded papers into Qdrant...")
        embed_model = LangchainEmbedding(
            HuggingFaceEmbedding(model_name=config["embedding_model"])
        )
        llm = Ollama(model=config["llm_name"], base_url=config["llm_url"])
        data = Data(config)
        index = data.ingest(embedder=embed_model, llm=llm)
        st.success("Ingestion to Qdrant Complete!")
        st.write(f"Data indexed successfully to Qdrant Collection: {config['collection_name']}")

    # Additional Information
    st.write("\n")
    st.write("Once the papers are downloaded, you can also perform search queries on the Qdrant database.")

if __name__ == "__main__":
    main()
