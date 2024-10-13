import json
from extractor import extract_multi_needle
from extractor.schema import TechCompany
from extractor.utils import json_to_csv

def main():
    example_needles = [
        "Ryoshi, based in Neo Tokyo, Japan, is a private quantum computing firm founded in 2031, currently valued at $8.7 billion with 1,200 employees focused on quantum cryptography."
    ]

    # Load the haystack text from a JSON file
    with open('/Users/siddartha/Desktop/github/Needle-In-Haystack/haystack.txt', 'r') as f:
        haystack_text = f.read()
    # Run the extraction
    extracted_data = extract_multi_needle(schema=TechCompany, haystack=haystack_text, example_needles=example_needles)

    # Serialize the extracted data to a JSON file
    with open('extracted_needles.json', 'w') as f:
        json.dump([item.dict() for item in extracted_data], f, indent=2)

    print("\nExtraction complete. Results saved to 'extracted_needles.json'.")

    # convert JSON to CSV
    json_to_csv('extracted_needles.json', 'extracted_needles.csv')

if __name__ == '__main__':
    main()
