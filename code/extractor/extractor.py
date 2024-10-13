import os
import re
import numpy as np
from typing import Type, List, TypeVar
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
load_dotenv()

from .utils import (
    generate_schema_description,
    generate_keywords,
    process_sentences_in_parallel
)

T = TypeVar('T', bound=BaseModel)

def extract_multi_needle(schema: Type[T], haystack: str, example_needles: List[str]) -> List[T]:
    """
    Extracts information from a large text (haystack) based on example sentences (needles)
    and a defined schema. Returns a list of extracted data conforming to the schema.
    """
    # Initialize the SentenceTransformer model for embeddings (fast and efficient)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Split the haystack into individual sentences using regex
    sentences = re.split(r'(?<=[.!?])\s+', haystack)

    # Compute embeddings for the sentences in the haystack
    sentence_embeddings = embedding_model.encode(
        sentences, batch_size=256, show_progress_bar=True
    )

    # Compute embeddings for the example needles
    example_embeddings = embedding_model.encode(
        example_needles, batch_size=256, show_progress_bar=True
    )

    # Normalize embeddings to unit vectors for cosine similarity calculation
    sentence_embeddings_normalized = sentence_embeddings / np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
    example_embeddings_normalized = example_embeddings / np.linalg.norm(example_embeddings, axis=1, keepdims=True)

    # Compute cosine similarities between example needles and sentences
    cosine_similarities = np.dot(example_embeddings_normalized, sentence_embeddings_normalized.T)

    # Set a similarity threshold to select relevant sentences
    similarity_threshold = 0.3  # Adjust this value as needed

    # Get indices of sentences that have similarity above the threshold
    candidate_indices = np.argwhere(cosine_similarities >= similarity_threshold)[:, 1]

    # Retrieve the candidate sentences based on the indices
    candidate_sentences = [sentences[idx] for idx in set(candidate_indices)]

    # Initialize the Azure OpenAI LLM model
    model = AzureChatOpenAI(
        openai_api_version=os.environ.get("AZURE_OPENAI_VERSION", "2023-03-15-preview"),
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
        azure_endpoint=os.environ.get(
            "AZURE_OPENAI_ENDPOINT",
            "https://your-openai-endpoint.azure.com"
        ),
        openai_api_key=os.environ.get("AZURE_OPENAI_KEY", "api_key"),
    )

    # Generate a description of the schema to include in the prompts
    schema_description = generate_schema_description(schema)

    # Generate keywords using the LLM
    keywords = generate_keywords(example_needles, schema_description, model)

    # Include sentences that contain any of the generated keywords
    keyword_sentences = [
        sentence for sentence in sentences
        if any(keyword.lower() in sentence.lower() for keyword in keywords)
    ]

    # Combine the candidate sentences from embeddings and keyword matching
    candidate_sentences = list(set(candidate_sentences).union(set(keyword_sentences)))

    print(f"Number of candidate sentences: {len(candidate_sentences)}")

    # Construct the system prompt with schema description for the LLM
    system_prompt = f"""
You are an assistant that extracts information from text according to a given schema.

The schema is:
{schema_description}

Your task is to read the provided text and extract any information that matches the schema.

Provide the extracted data as a JSON object conforming to the schema.

If the text does not contain relevant information, output an empty JSON object.

Only provide the JSON object, and no additional text.

Consider variations in sentence structure and wording. Extract information even if the text differs from the examples.
"""

    # Process the candidate sentences in parallel using threading
    extracted_needles = process_sentences_in_parallel(candidate_sentences, system_prompt, model, schema)

    return extracted_needles
