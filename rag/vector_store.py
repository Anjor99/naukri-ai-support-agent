from chromadb import Collection, PersistentClient
from .chunking import Chunk, ChunkingStrategy
from .embeddings import EmbeddingModel
import numpy as np

class VectorStore:
    def __init__(self, embedding_model: EmbeddingModel):
        # initialize persistent chromadb client and collection
        self.client = PersistentClient(path="data/chroma_db")
        self.embedding_model = embedding_model
        
    def create_collection(self, collection_name: str) -> Collection:
        # Create a new collection in ChromaDB or retrieve an existing one
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_chunks(self, collection: Collection, chunks: list[Chunk]):
        # Prepare data for insertion
        ids = [chunk.chunk_id for chunk in chunks]
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.embed_texts(texts)
        doc_ids = [chunk.doc_id for chunk in chunks]
        strategies = [chunk.strategy.value for chunk in chunks]
        
        # Insert into the collection
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),  # Convert numpy array to list for ChromaDB
            metadatas=[{"doc_id": doc_id, "strategy": strategy} for doc_id, strategy in zip(doc_ids, strategies)]
        )
        
    def search(self, collection: Collection, query: str, top_k: int = 5):
        # Change into using cosine similarity search with the embedding model
        query_embedding = self.embedding_model.embed_text(query)
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        return results
    
    def delete_collection(self, collection_name: str):
        # Delete a collection from ChromaDB
        self.client.delete_collection(name=collection_name)
        
        
# Testing the VectorStore class
if __name__ == "__main__":
    from .embeddings import EmbeddingModel
    from .loader import load_documents
    from .chunking import chunk_documents
    
    # Initialize embedding model and vector store
    embedding_model = EmbeddingModel()
    vector_store = VectorStore(embedding_model)
    
    # Load documents and create chunks
    documents = load_documents("knowledge_base")
    chunks = chunk_documents(documents, strategy=ChunkingStrategy.SENTENCE_BASED)
    
    # Create a collection and add chunks to it
    collection = vector_store.create_collection("sentence_based_collection")
    vector_store.add_chunks(collection, chunks)
    
    # Perform a search query
    query = "remote work"
    results = vector_store.search(collection, query, top_k=3)
    
    print(f"Search results for query '{query}':")
    # Readable output of results
    for i in range(len(results['ids'][0])):
        print(f"Result {i+1}:")
        print(f"  ID: {results['ids'][0][i]}")
        print(f"  Document: {results['documents'][0][i]}")
        print(f"  Metadata: {results['metadatas'][0][i]}")
        print(f"  Distance: {results['distances'][0][i]}")