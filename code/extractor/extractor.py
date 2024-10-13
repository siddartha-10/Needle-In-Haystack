import os
import re
import numpy as np
from typing import Type, List, TypeVar
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
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
    # Sentence Transformer model used for computing embeddings
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Split the haystack into individual sentences using regex
    sentences = re.split(r'(?<=[.!?])\s+', haystack)

    # Computing embeddings for the sentences
    sentence_embeddings = embedding_model.encode(
        sentences, batch_size=256, show_progress_bar=True
    )

    # Compute embeddings for the example needles
    example_embeddings = embedding_model.encode(
        example_needles, batch_size=256, show_progress_bar=True
    )

    # Normalize embeddings for simple cosine similarity computation
    sentence_embeddings_normalized = sentence_embeddings / np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
    example_embeddings_normalized = example_embeddings / np.linalg.norm(example_embeddings, axis=1, keepdims=True)

    # Compute cosine similarities
    cosine_similarities = np.dot(example_embeddings_normalized, sentence_embeddings_normalized.T)

    # Setting a threshold for cosine similarity
    similarity_threshold = 0.3

    candidate_indices = np.argwhere(cosine_similarities >= similarity_threshold)[:, 1]

    # Retrieve the candidate sentences based on the indices
    candidate_sentences = [sentences[idx] for idx in set(candidate_indices)]

    model = ChatOpenAI(model = 'gpt-4o-mini')

    # Generating the schema description
    schema_description = generate_schema_description(schema)

    # Generating keywords using the LLM
    keywords = generate_keywords(example_needles, schema_description, model)

    # Include sentences that contain any of the generated keywords
    keyword_sentences = [
        sentence for sentence in sentences
        if any(keyword.lower() in sentence.lower() for keyword in keywords)
    ]

    # We want to include all candidate sentences and keyword sentences, because of the possibility of false negatives
    candidate_sentences = list(set(candidate_sentences).union(set(keyword_sentences)))

    print(f"Number of candidate sentences: {len(candidate_sentences)}")

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
