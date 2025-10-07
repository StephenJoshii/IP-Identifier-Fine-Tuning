import os
import json
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Make sure it's in your .env file.")

client = OpenAI(api_key=api_key)


fine_tuned_model_id = "ft:gpt-3.5-turbo-0125:personal::COA4j20L"


test_file_name = "dataset_test.jsonl"
# ------------------------------------



true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0
total_tested = 0
# ---------------------------------



print(f"Starting evaluation with model: {fine_tuned_model_id}")

try:
    with open(test_file_name, 'r') as f:
        for line in f:
            total_tested += 1
            entry = json.loads(line)
            

            system_message = entry['messages'][0] # The system prompt
            user_message = entry['messages'][1]   # The user's input
            actual_label = entry['messages'][2]['content'].strip()
            

            response = client.chat.completions.create(
                model=fine_tuned_model_id,
                messages=[system_message, user_message], 
                max_tokens=5, 
                temperature=0 
            )
            
            prediction = response.choices[0].message.content.strip()
            
            # Compare prediction to the actual label and update counters
            if prediction == "Positive" and actual_label == "Positive":
                true_positives += 1
            elif prediction == "Negative" and actual_label == "Negative":
                true_negatives += 1
            elif prediction == "Positive" and actual_label == "Negative":
                false_positives += 1
            elif prediction == "Negative" and actual_label == "Positive":
                false_negatives += 1


    print("\n--- Evaluation Complete ---")
    print(f"Total Samples Tested: {total_tested}")

    print("\n--- Confusion Matrix ---")
    print(f"True Positives (TP): {true_positives}")
    print(f"True Negatives (TN): {true_negatives}")
    print(f"False Positives (FP): {false_positives}")
    print(f"False Negatives (FN): {false_negatives}")
    
    # Calculate metrics
    accuracy = (true_positives + true_negatives) / total_tested if total_tested > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n--- Performance Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print("---------------------------\n")

except FileNotFoundError:
    print(f"Error: The test file '{test_file_name}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
# ------------------------------------