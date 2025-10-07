import os
from dotenv import load_dotenv
from openai import OpenAI


print(f"Successfully found and loaded .env file: {load_dotenv()}")


api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Check your .env file content and location.")


client = OpenAI(api_key=api_key)


job_id = "ftjob-UxN2aJPlrJEHIwPW0m2MBOmM" 

print(f"Checking status for job: {job_id}")
try:
    job = client.fine_tuning.jobs.retrieve(job_id)

    print(f"Status: {job.status}")
    if job.fine_tuned_model:
        print(f"Fine-tuned model ID: {job.fine_tuned_model}")
    else:
        print("Fine-tuned model ID is not available yet.")

except Exception as e:
    print(f"An error occurred: {e}")