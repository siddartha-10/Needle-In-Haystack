import sys
import os
sys.path.append('/Users/siddartha/Desktop/github/Needle-In-Haystack/code') 

import json
from extractor import extract_multi_needle
from extractor_1.schema import Football
from extractor.utils import json_to_csv

def main():
    example_needles = [
        "Ajax, based in Amsterdam, Netherlands, was established in 1900 and valued at $1.2 billion, with Johan Cruyff as their greatest player."
    ]

    # Load the haystack text from a JSON file
    with open('/Users/siddartha/Desktop/github/Needle-In-Haystack/Experiment/haystack_exp.txt', 'r') as f:
        haystack_text = f.read()
    # Run the extraction
    extracted_data = extract_multi_needle(schema=Football, haystack=haystack_text, example_needles=example_needles)

    # Serialize the extracted data to a JSON file
    with open('extracted_needles.json', 'w') as f:
        json.dump([item.dict() for item in extracted_data], f, indent=2)

    print("\nExtraction complete. Results saved to 'extracted_needles.json'.")

    # convert JSON to CSV
    json_to_csv('extracted_needles.json', 'extracted_needles.csv')

if __name__ == '__main__':
    main()
