from openai import OpenAI
import os


api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Make sure it's in your .env file.")

# Use the key to create the client
client = OpenAI(api_key=api_key)

training_file_name = "dataset_train.jsonl"

print(f"Uploading file: {training_file_name}")
try:
    training_file = client.files.create(
      file=open(training_file_name, "rb"),

      purpose="fine-tune"
    )

    print("File uploaded successfully:")
    print(f"File ID: {training_file.id}")

except FileNotFoundError:
    print(f"Error: The file '{training_file_name}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")