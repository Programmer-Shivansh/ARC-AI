import json
score = 0 
correct = 0 
def load_json(file_path):
    """Loads a JSON file and returns its content."""
    with open(file_path, 'r') as file:
        return json.load(file)

def process_and_print_mapped_cases(json1, json2):
    """Processes and prints train cases along with mapped outputs."""
    for case_id, case_data in json1.items():
        print(f"\nCase ID: {case_id}")
        
        # Process train cases
        if "train" in case_data:
            for i, train_case in enumerate(case_data["train"]):
                input_data = train_case.get("input", [])
                output_data = train_case.get("output", [])
                
                print(f"\n  Train Case {i+1}:")
                print(f"    Input: {input_data}")
                print(f"    Expected Output: {output_data}")


def score_calculator(json1, json2):    
    for case_id, case_data in json1.items():
        score += 1
        print(f"\nCase ID: {case_id}")  
        if "test" in case_data:
                # Attempt to retrieve the mapped output from JSON2
                test_case = case_data["test"]
                # print(test_case)
                input_datas = test_case[0].get("input", [])
                mapped_outputs = json2.get(case_id, [])
                if input_datas == mapped_outputs:
                    correct += 1
                # print(f"\n  Test Case :")
                # print(f"    Input: {input_datas}")
                # print(f"    Expected Output: {mapped_outputs}")

# Paths to your JSON files
json_file_path_1 = "Dataset/arc-agi_evaluation_challenges.json"  # Replace with your first JSON file path
json_file_path_2 = "Dataset/arc-agi_evaluation_solutions.json"  # Replace with your second JSON file path

# Load JSON files
json1 = load_json(json_file_path_1)
json2 = load_json(json_file_path_2)

# Process and print mapped cases
process_and_print_mapped_cases(json1, json2)

