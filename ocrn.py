import os
import json
import requests
import random
import string

# Replace 'your_api_key' with your actual API key
api_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiYTMyZGQyMTAtNzY0Ny00ZDg3LTk2ZjYtOGM4MDIxOTc3OTM0IiwidHlwZSI6ImFwaV90b2tlbiJ9.eRDTpdfJtrHYFytPrLZdE7WOYZE8TaRiSb3eAfKsgy4'
api_url = 'https://api.edenai.run/v2/ocr/ocr'

# Folder containing the images
folder_path = r'E:\helmet\hnp1\numberplate'

# Files to store previously assigned mappings and recognized plates
mapping_file = 'number_plate_mapping.json'
output_path = 'extracted_text.json'

# Load existing mappings if available
if os.path.exists(mapping_file) and os.path.getsize(mapping_file) > 0:
    with open(mapping_file, 'r') as f:
        number_mapping = json.load(f)
else:
    number_mapping = {}

# Load existing extracted text results safely
if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
    try:
        with open(output_path, 'r') as f:
            existing_results = json.load(f)
    except json.JSONDecodeError:
        existing_results = {}  # Reset if file is corrupted
else:
    existing_results = {}

# Function to perform OCR on a single image
def perform_ocr(image_path):
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
    
    headers = {
        'Authorization': f'Bearer {api_key}'
    }
    files = {
        'file': (os.path.basename(image_path), image_data, 'image/jpeg')
    }
    data = {
        'providers': ['api4ai']
    }

    response = requests.post(api_url, headers=headers, files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        extracted_text = result['api4ai']['text']
        
        # Replace '\n' with a space
        extracted_text = extracted_text.replace('\n', ' ')
        
        # Check if extracted text contains only numbers
        if extracted_text.isdigit():
            if extracted_text in number_mapping:
                extracted_text = number_mapping[extracted_text] + " " + extracted_text
            else:
                random_letters = ''.join(random.choices(string.ascii_uppercase, k=2))
                number_mapping[extracted_text] = random_letters
                extracted_text = random_letters + " " + extracted_text
        
        return extracted_text
    else:
        print('Error:', response.status_code, response.text)
        return None

# Dictionary to store new results
results = {}

# Process each image in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(folder_path, filename)
        print(f'Processing {filename}...')
        text = perform_ocr(image_path)
        
        # Only store new number plates
        if text is not None and text not in existing_results.values():
            results[filename] = text

# Save the updated results to a JSON file
if results:
    existing_results.update(results)
    with open(output_path, 'w') as json_file:
        json.dump(existing_results, json_file, indent=4)

# Save the updated number mapping
with open(mapping_file, 'w') as f:
    json.dump(number_mapping, f, indent=4)

print(f'OCR results saved to {output_path}')
