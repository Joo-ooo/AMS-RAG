# main.py

from graph_pipeline import (
    ConfigManager,
    ModelRegistry,
    GraphIngestionPipeline,
    GraphQueryPipeline,
)

def run_pipeline():
    """
    Initializes and runs the graph query pipeline for a hardcoded question.
    """
    # --- Configuration ---
    DOTENV_PATH = "../.env"

    REBUILD_INDEX = False

    USER_QUESTION = "What partnerships exist in academia?"
    # --- End of Configuration ---

    # Initialize Configuration and Models
    print("Initializing configuration and models...")
    try:
        config = ConfigManager(DOTENV_PATH)
        models = ModelRegistry(config)
    except ValueError as e:
        print(f"Error during initialization: {e}")
        print(f"Please check that your .env file is correctly located at '{DOTENV_PATH}' and contains all required variables.")
        return
    print("Initialization complete.")

    if REBUILD_INDEX:
        print("\n--- Starting Index Build ---")
        ingestion_pipeline = GraphIngestionPipeline(config, models)

        # Pass the directory directly to the pipeline
        ingestion_pipeline.ingest_from_directory(config.data_directory)
        
        print("--- Ingestion Process Complete ---\n")

    # Initialize Query Pipeline and Run the Hardcoded Question
    print(f"\n--- Running Query ---\nQuestion: {USER_QUESTION}")
    try:
        query_pipeline = GraphQueryPipeline(config, models)
        response = query_pipeline.query(USER_QUESTION, use_llm="openai")
        
        print("\n--- Response ---")
        print(response)
        print("----------------")
        
    except Exception as e:
        print(f"\nAn error occurred during the query process: {e}")
        print("Please ensure the graph index has been built correctly in Neo4j.")
        if not REBUILD_INDEX:
            print("You may need to set REBUILD_INDEX = True in this script and run it again.")


if __name__ == "__main__":
    run_pipeline()
