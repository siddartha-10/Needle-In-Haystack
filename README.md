# Needle in a Haystack Extraction Tool

## Overview

This project provides a Python module that extracts specific information (the "needles") from a large body of text (the "haystack") using example sentences and a defined schema. It leverages advanced natural language processing techniques, including embeddings and language models, to efficiently find and extract relevant data and store it a CSV file.

## Features

- **Embeddings for Similarity Matching**: Uses SentenceTransformer models to compute embeddings and identify sentences similar to the provided examples.
- **Keyword Generation**: Dynamically generates keywords based on example needles and schema to enhance sentence selection.
- **Parallel Processing**: Implements multithreading to process API calls concurrently, significantly improving performance.
- **Schema-Based Extraction**: Utilizes Pydantic schemas to define the structure of the extracted data.
- **Modular and Readable Code**: Organized into functions with clear comments for easy understanding and maintenance.

## Prerequisites

- Python 3.12
- OpenAI API Key
  
## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/siddartha-10/needle-in-haystack.git
   cd needle-in-haystack

2. **Setup env variables**

   ```bash
   OPENAI_API_KEY = "your_api_key"
   
3. **Installing the Requirements**

   ```bash
   pip install -r requirements.txt 

## Code Explanation

Here is the basic overview of how the code works.

```bash
    1. Split haystack into sentences.
    2. Compute embeddings for sentences and example needles.
    3. Find candidate sentences based on similarity.
    4. Generate keywords using the LLM.
    5. Find additional candidate sentences containing keywords.
    6. Process candidate sentences in parallel to extract data.
    7. Return a list of extracted data conforming to the schema.
