import os
import json
import numpy as np
import concurrent.futures
import re
import csv
from typing import Type, List, TypeVar
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from tqdm import tqdm  # For progress bar visualization

T = TypeVar('T', bound=BaseModel)

def generate_schema_description(schema: Type[BaseModel]) -> str:
    """
    Generates a text description of the schema fields and their types.
    """
    schema_description = ""
    for field_name, field in schema.__fields__.items():
        field_desc = field.description or ''
        field_type = (
            field.annotation.__name__ if hasattr(field.annotation, '__name__') else str(field.annotation)
        )
        schema_description += f"- {field_name} ({field_type}): {field_desc}\n"
    return schema_description

def generate_keywords(example_needles: List[str], schema_description: str, model) -> List[str]:
    """
    Uses the LLM to generate a list of keywords based on the example needles and schema.
    """
    # Construct a prompt for the LLM
    prompt = f"""Given the following schema and example needles, generate a list of keywords that would be useful for identifying relevant sentences in a text. The keywords should be related to the schema fields and the type of information we're looking for.

Schema:
{schema_description}

Example needles:
{', '.join(example_needles)}

Please provide a comma-separated list of approximately 10 keywords. These keywords should be closed words in English, i.e., there are needles present in the haystack which are structurally very similar to the example needles. Choose keywords that will help identify these other similar needles as well."""
    # Create the conversation messages for the LLM
    messages = [
        SystemMessage(content="You are an assistant that extracts keywords from text."),
        HumanMessage(content=prompt)
    ]

    # Call the LLM to generate keywords
    response = model.invoke(messages, temperature=0.3)

    # Parse the response to extract keywords
    keywords_text = response.content.strip()
    keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
    return keywords

def process_sentences_in_parallel(candidate_sentences: List[str], system_prompt: str, model, schema: Type[T]) -> List[T]:
    """
    Processes multiple sentences in parallel using threading to make API calls concurrently.
    Returns a list of extracted data conforming to the schema.
    """
    results = {}
    error_indices = []

    # Use ThreadPoolExecutor to process sentences concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks to the executor
        future_to_index = {
            executor.submit(process_sentence, index, text, system_prompt, model, schema): index
            for index, text in enumerate(candidate_sentences)
        }

        # Process the futures as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_index),
                           total=len(candidate_sentences), desc="Processing Sentences"):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
            except Exception as exc:
                print(f'Sentence {index} generated an exception: {exc}')
                results[index] = None
                error_indices.append(index)

    # Collect the extracted items from results
    extracted_items = [result for result in results.values() if result is not None]

    return extracted_items

def process_sentence(index: int, text: str, system_prompt: str, model, schema: Type[T]) -> T:
    """
    Processes a single sentence using the LLM to extract information according to the schema.
    Returns an instance of the schema if data is extracted, or None otherwise.
    """
    # Create the conversation messages for the LLM
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=text)
    ]

    # Call the LLM to process the text
    response = model.invoke(messages, temperature=0.6)

    # Attempt to parse the LLM response as JSON
    try:
        data = json.loads(response.content)
        if data:  # If data is not empty
            # Validate and instantiate the schema with the extracted data
            item = schema(**data)
            return item
    except json.JSONDecodeError:
        print(f"JSONDecodeError for sentence {index}: {text}")
        print(f"LLM response: {response.content}")
    except Exception as e:
        print(f"Exception for sentence {index}: {text}")
        print(f"Error: {e}")
    return None

def json_to_csv(json_file: str, csv_file: str):
    """
    Converts a JSON file to a CSV file.
    """
    # Check if the JSON file exists
    if not os.path.exists(json_file):
        print(f"File {json_file} not found.")
        return

    # Read the JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # If the JSON data is a dictionary, convert it to a list of dictionaries
    if isinstance(data, dict):
        data = [data]

    # Get the keys (column names) from the first item
    fieldnames = data[0].keys()

    # Write data to the CSV file
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"CSV file saved as {csv_file}")
