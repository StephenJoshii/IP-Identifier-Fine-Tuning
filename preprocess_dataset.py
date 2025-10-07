import pandas as pd
import json
from sklearn.model_selection import train_test_split


try:
    df = pd.read_csv('dataset_small.csv')
except FileNotFoundError:
    print("Error: 'dataset_small.csv' not found. Make sure it's in the same directory.")
    exit()

def create_jsonl_entry(row):

    system_message = "You are a helpful assistant that determines if a string is a real IP address or not."
    

    user_content = f"{row['text']} ; {row['ip']}"
    

    assistant_content = "Negative" if row['class'] == 'N' else "Positive"
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]
    return {"messages": messages}


jsonl_data = df.apply(create_jsonl_entry, axis=1).tolist()


train_data, test_data = train_test_split(jsonl_data, test_size=0.2, random_state=42)


def save_to_jsonl(data, filename):
    with open(filename, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

save_to_jsonl(train_data, 'dataset_train.jsonl')
save_to_jsonl(test_data, 'dataset_test.jsonl')

print("Successfully created 'dataset_train.jsonl' and 'dataset_test.jsonl' in the new format.")