from rag_pipeline import ConfigManager, ModelProvider, DocumentIngestor, RAGPipeline

def main():
    # Initialize Configs
    try:
        config = ConfigManager()
        models = ModelProvider(config)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error during initialization: {e}")
        return

    # Ingest Documents
    # This only needs to be run once or when documents change
    # The resulting `documents` object can be saved and loaded to speed up subsequent runs
    ingestor = DocumentIngestor(models)
    documents = ingestor.ingest_pdfs_from_directory(config.data_directory)
    
    if not documents:
        print("No documents were ingested. Exiting.")
        return

    # Setup and Run the RAG Pipeline
    pipeline = RAGPipeline(config, models)
    pipeline.setup_pipeline(documents)

    # Query the pipeline
    query_text = "What percentage of reported crimes in 2020 were scams?"
    result = pipeline.query(query_text)
    
    print("\n--- Query Result ---")
    print(f"Answer: {result['answer']}")
    print("\n--- Top Source Node Content ---")
    if result['source_nodes']:
        print(result['source_nodes'][0].get_content())
    print("---------------------\n")


if __name__ == "__main__":
    main()
