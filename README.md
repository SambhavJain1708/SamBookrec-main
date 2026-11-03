# SamBookrec

# AI-Assisted Book Recommender

A simple Python-based semantic search application that recommends books based on the meaning and context of a user's query rather than relying on keyword matching.

The application uses Sentence Transformers to generate text embeddings and FAISS to find semantically similar books.

## Features

- Semantic search based on natural language queries.
- Book recommendations using vector similarity.
- Sentence Transformers for text embeddings.
- FAISS for efficient similarity search.
- CSV-based book dataset.
- Function-based Python implementation (no classes).

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Dataset management |
| NumPy | Numerical operations |
| Sentence Transformers | Generate text embeddings |
| FAISS | Vector similarity search |

## Project Structure

```text
book-recommender/
│
├── data/
│   └── books.csv
│
├── app.py
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.9 or higher
- pip
- Internet connection for downloading the embedding model initially

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd book-recommender
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Dependencies

Create a `requirements.txt` file:

```text
pandas
numpy
sentence-transformers
faiss-cpu
```

## Dataset

The application uses a CSV file containing book information.

Location:

```text
data/books.csv
```

Expected columns:

| Column | Description |
|---|---|
| title | Book title |
| author | Book author |
| description | Book description |

Example:

```csv
title,author,description
Atomic Habits,James Clear,"A practical guide to building better habits."
Deep Work,Cal Newport,"A guide to improving focus and productivity."
Thinking Fast and Slow,Daniel Kahneman,"Explores human psychology and decision making."
```

## How It Works

The application follows these steps:

### 1. Load Books

The `load_books()` function reads the CSV dataset using Pandas and combines the book title, author and description.

### 2. Generate Embeddings

The `generate_embeddings()` function converts book descriptions into numerical vectors using the Sentence Transformer model.

Model:

```text
all-MiniLM-L6-v2
```

Each book is represented by a 384-dimensional vector.

### 3. Create FAISS Index

The `create_faiss_index()` function creates a FAISS index and stores the book embeddings.

It uses:

```python
faiss.IndexFlatIP()
```

Vectors are normalized before indexing, allowing inner product search to calculate cosine similarity.

### 4. Semantic Search

The `search_books()` function:

- Accepts a natural language query.
- Converts the query into an embedding.
- Searches the FAISS index.
- Retrieves the most similar books.
- Returns ranked recommendations with similarity scores.

### 5. Display Recommendations

The `display_recommendations()` function displays recommended books, authors, descriptions and similarity scores.

## Application Execution

Run the application:

```bash
python app.py
```

Example:

```text
AI Book Recommender
-------------------

What kind of book are you looking for?

I want to improve my decision-making
and understand human psychology.

Recommended Books:

1. Thinking Fast and Slow
   Author: Daniel Kahneman
   Similarity: 0.82

2. The Art of Thinking Clearly
   Author: Rolf Dobelli
   Similarity: 0.76
```

*Note: Similarity scores above are illustrative.*

## Architecture / Data Flow

```text
              User Query
                  |
                  v
           Query Embedding
                  |
                  v
            FAISS Search
                  |
                  v
          Similarity Ranking
                  |
                  v
          Retrieve Book Data
                  |
                  v
        Recommended Books
```

### Function Overview

| Function | Responsibility |
|---|---|
| `load_books()` | Load book dataset |
| `generate_embeddings()` | Generate book embeddings |
| `create_faiss_index()` | Create vector index |
| `search_books()` | Perform semantic search |
| `display_recommendations()` | Display recommendations |
| `main()` | Application entry point |

## Future Enhancements

- Integrate an LLM to explain recommendations.
- Add personalized recommendations based on reading history.
- Introduce genre and author filters.
- Store embeddings and FAISS indexes on disk.
- Replace CSV with a database.
- Build a Streamlit-based web interface.
- Add user ratings and feedback.
- Introduce hybrid search combining semantic and keyword search.

## License

This project is intended for educational and learning purposes.
