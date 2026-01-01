from vector_pipeline import ConfigManager, ModelProvider, DocumentIngestor, RAGPipeline
import os

def main():
    # Initialize Configs
    try:
        config = ConfigManager()
        models = ModelProvider(config)
        ingestor = DocumentIngestor(models)
        pipeline = RAGPipeline(config, models)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error during initialization: {e}")
        return

    print("Checking for new files...")
    new_file_paths = ingestor.get_new_file_paths(config.data_directory)

    documents = []
    if new_file_paths:
        print(f"Found {len(new_file_paths)} new files. Running OCR...")
        # Run OCR ONLY on the new files
        documents = ingestor.ingest_pdfs_from_directory(
            config.data_directory,
            specific_files=new_file_paths  # Pass the filtered list
        )
    else:
        print("All files are already processed. Skipping OCR.")

    pipeline.setup_pipeline(documents)

    # Query the pipeline
    query_text = "What percentage of reported crimes in 2020 were scams?"
    result = pipeline.query_engine.query(query_text)
    
    print("\n--- Query Result ---")
    print(f"Answer: {result.response}")
    # print("\n--- Top Source Node Content ---")
    # if result.source_nodes:
    #     print(result.source_nodes[0].get_content())
    # print("---------------------\n")


if __name__ == "__main__":
    main()
