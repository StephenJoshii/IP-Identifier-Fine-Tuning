from openai import OpenAI
import os


api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Make sure it's in your .env file.")

# Use the key to create the client
client = OpenAI(api_key=api_key)

file_id = "file-1H1gJmRTsKjqBz45Gcj1p2"

print("Starting fine-tuning job...")
try:
    job = client.fine_tuning.jobs.create(
        training_file=file_id,

        model="gpt-3.5-turbo" 
    )

    print("Job created successfully:")
    print(job)

except Exception as e:
    print(f"An error occurred: {e}")