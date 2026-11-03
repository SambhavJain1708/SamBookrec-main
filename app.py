
import pandas as pd
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


# 1. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 2. Load book dataset
def load_books(file_path):

    books = pd.read_csv(file_path)

    # Combine book information
    books["content"] = (
        books["title"] + ". " +
        books["author"] + ". " +
        books["description"]
    )

    return books


# 3. Generate book embeddings
def generate_embeddings(books):

    descriptions = books["content"].tolist()

    embeddings = model.encode(
        descriptions,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    return embeddings


# 4. Create FAISS index
def create_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# 5. Search books semantically
def search_books(query, books, index, top_k=5):

    # Generate query embedding
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    query_embedding = query_embedding.astype("float32")

    faiss.normalize_L2(query_embedding)

    # Perform similarity search
    scores, indices = index.search(
        query_embedding,
        min(top_k, len(books))
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        book = books.iloc[idx]

        results.append({
            "title": book["title"],
            "author": book["author"],
            "description": book["description"],
            "similarity": round(float(score), 3)
        })

    return results


# 6. Display recommendations
def display_recommendations(results):

    print("\nRecommended Books")
    print("=================\n")

    for number, book in enumerate(results, 1):

        print(f"{number}. {book['title']}")
        print(f"   Author: {book['author']}")
        print(f"   Similarity: {book['similarity']}")
        print(f"   Description: {book['description']}")
        print()


# 7. Main application
def main():

    books = load_books("data/books.csv")

    embeddings = generate_embeddings(books)

    index = create_faiss_index(embeddings)

    print("AI Book Recommender")
    print("-------------------")

    while True:

        query = input(
            "\nWhat kind of book are you looking for? "
        )

        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        results = search_books(
            query,
            books,
            index,
            top_k=2
        )

        display_recommendations(results)


if __name__ == "__main__":
    main()
