
import pandas as pd
import chromadb

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

    return model.encode(descriptions, convert_to_numpy=True)


# 4. Create ChromaDB collection
def create_chroma_collection(embeddings):

    client = chromadb.Client()
    collection = client.create_collection(
        name="books",
        metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[str(index) for index in range(len(embeddings))],
        embeddings=embeddings.tolist()
    )

    return collection


# 5. Search books semantically
def search_books(query, books, collection, top_k=5):

    # Generate query embedding
    query_embedding = model.encode([query], convert_to_numpy=True)

    # Perform similarity search
    matches = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=min(top_k, len(books))
    )

    results = []

    for book_id, distance in zip(matches["ids"][0], matches["distances"][0]):
        book = books.iloc[int(book_id)]

        results.append({
            "title": book["title"],
            "author": book["author"],
            "description": book["description"],
            "similarity": round(1 - float(distance), 3)
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

    collection = create_chroma_collection(embeddings)

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
            collection,
            top_k=2
        )

        display_recommendations(results)


if __name__ == "__main__":
    main()
