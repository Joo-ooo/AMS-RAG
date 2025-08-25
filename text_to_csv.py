import pandas as pd

column_names = ["question", "answer", "context"]

try:
    # The only change is here: 'data.csv' is now 'data.txt'
    df = pd.read_csv('sql_benchmark.txt', header=None, names=column_names)

    # This part remains the same
    print("--- DataFrame View ---")
    print(df)
    
    df.to_csv('sql_benchmark.csv', index=False)
    print("\nSuccessfully read from sql_benchmark.txt and created sql_benchmark.csv.")

except FileNotFoundError:
    print("Error: sql_benchmark.txt not found. Please make sure the file exists.")
