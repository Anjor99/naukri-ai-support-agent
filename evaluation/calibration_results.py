# 1. Import
from rag.loader import load_documents
from rag.chunking import fixed_size_chunks
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore

# 2. Define your calibration queries
in_scope_queries = [
    "What is the notice period policy?",
    "How are interviews scheduled?",
    "Who is eligible for remote work?",
    "What is the referral bonus policy?",
    "What happens during background verification?"
]

out_of_scope_queries = [
    "How do I make chocolate cake?",
    "What is the capital of France?",
    "How to play the guitar?",
    "What is the weather like today?",
]

# 3. Load your existing cosine Chroma collection
emb_model = EmbeddingModel()
vector_store = VectorStore(emb_model)
collection = vector_store.create_collection("fixed_size_collection")

# populate the collection with chunks from your knowledge base
documents =  load_documents("knowledge_base")
chunks = fixed_size_chunks(documents)
vector_store.add_chunks(collection, chunks)

# 4. For every query:
#    a. search(collection, query, top_k=1)
#    b. get the returned distance
#    c. similarity = 1 - distance
#    d. get the returned document/chunk ID
#    e. store:
#       query, scope, document_id, similarity
table_data = []

for query in in_scope_queries:
    results = vector_store.search(collection, query, top_k=1)
    distance = results['distances'][0][0]
    similarity = 1 - distance
    doc_id = results['ids'][0][0]
    table_data.append((query, "in-scope", doc_id, similarity))
    
for query in out_of_scope_queries:
    results = vector_store.search(collection, query, top_k=1)
    distance = results['distances'][0][0]
    similarity = 1 - distance
    doc_id = results['ids'][0][0]
    table_data.append((query, "out-of-scope", doc_id, similarity))

# 5. Print a Markdown table in readable format
print("| Query | Scope | Document ID | Similarity |")
print("|-------|-------|-------------|------------|")
for row in table_data:
    print(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]:.4f} |")


# For each result:
# print a row in the same format

# 6. Print the similarity ranges
#    in_scope_min / in_scope_max
#    out_scope_min / out_scope_max
in_scope_similarities = [row[3] for row in table_data if row[1] == "in-scope"]
out_of_scope_similarities = [row[3] for row in table_data if row[1] == "out-of-scope"]
print(f"\nIn-scope similarity range: {min(in_scope_similarities):.4f} - {max(in_scope_similarities):.4f}")
print(f"Out-of-scope similarity range: {min(out_of_scope_similarities):.4f} - {max(out_of_scope_similarities):.4f}")

# 7. Print the chosen threshold AFTER looking at the results.
# ./evaluation/calibration_results.md