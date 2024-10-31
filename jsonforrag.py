import pandas as pd
df = pd.read_csv("./dataset/spotifyreviewscombined.csv") # From Kaggle with more than 3 million data
df.fillna('NULL', inplace=True) 
df.replace('nan', '"NULL"', inplace=True)

data = df['combined_column'].astype(str).str.replace(': nan', ': "NaN"', regex=False)

context_list = data.apply(eval)

context_df = pd.DataFrame(context_list.tolist())  # Convert list of dicts to DataFrame

# Save DataFrame to JSON Lines format
context_df.to_json('./dataset/spotifyreviewscombined.jsonl', orient='records', lines=True)