from sql_pipeline import ConfigManager, DatabaseManager, AdvancedQueryEngine

def main():
    """
    Main function to initialize and run the ADVANCED SQL RAG pipeline.
    """

    # Initialize Configuration
    try:
        print("--- Initializing Pipeline ---")
        config = ConfigManager()
    except (FileNotFoundError, ValueError) as e:
        print(f"Configuration Error: {e}")
        return

    # Populate the Database
    db_manager = DatabaseManager(config=config)
    engine, all_table_names = db_manager.populate_database()

    # Set up the Advanced Query Engine
    query_engine_builder = AdvancedQueryEngine(engine, all_table_names, config)

    # Example Query
    query_str = "How many inmates had 'No Education' in 2022?"
    print("--- Executing Example Query ---")
    print(f"Question: {query_str}")

    response = query_engine_builder.query(query_str)

    # Print the prompt that was sent to the model
    if 'text_to_sql_prompt' in response.metadata:
        print(f"\n--- Prompt Sent to Model ---\n{response.metadata['text_to_sql_prompt']}")

    print(f"Generated SQL Query: {response.metadata['sql_query']}")

    # Extract the result from metadata
    sql_result = response.metadata.get('result')
    if sql_result:
        # Assuming the query returns a single values
        if len(sql_result) > 0 and len(sql_result[0]) > 0:
            final_response = sql_result[0][0]
        else:
            final_response = "No result found"
    else:
        final_response = response.response

    print(f"Response: {final_response}")
    print("--------------------------------\\n")

if __name__ == "__main__":
    main()
