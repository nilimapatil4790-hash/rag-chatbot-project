import json
import numpy as np
import faiss

# Load embeddings
with open("embeddings.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Convert embeddings into a NumPy array
embeddings = []

for item in data:
    vector = np.array(item["embedding"], dtype="float32")
    vector = vector.flatten()
    embeddings.append(vector)

embeddings = np.array(embeddings, dtype="float32")

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save the index
faiss.write_index(index, "vector_database.index")

# Save the text chunks
with open("chunks.json", "w", encoding="utf-8") as file:
    json.dump(
        [item["text"] for item in data],
        file,
        ensure_ascii=False,
        indent=2
    )

print("Vector database created successfully!")
print("Number of vectors:", index.ntotal)