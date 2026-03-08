from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

class VectorStore:

    def __init__(self):
        self.index = None
        self.text_chunks = []

    def build(self, text):
        splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_text(text)
        self.text_chunks = chunks

        embeddings = model.encode(chunks)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings))

    def search(self, query):
        query_vector = model.encode([query])
        distances, indices = self.index.search(query_vector, k=3)

        results = [self.text_chunks[i] for i in indices[0]]
        return results