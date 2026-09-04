import os
import pickle
import faiss

INDEX_PATH = "data/index/plantvillage.faiss"
RECORDS_PATH = "data/index/records.pkl"
VECTORIZER_PATH = "data/index/vectorizer.pkl"

class PlantVillageRetriever:
    def __init__(self):
        for path in (INDEX_PATH, RECORDS_PATH, VECTORIZER_PATH):
            if not os.path.exists(path):
                raise FileNotFoundError("RAG index missing. Run: python rag/build_index.py")
        self.index = faiss.read_index(INDEX_PATH)
        with open(RECORDS_PATH, "rb") as f:
            self.records = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            self.vectorizer = pickle.load(f)

    def search(self, crop, disease, symptoms=None, top_k=5):
        query = f"Crop: {crop} Disease: {disease} Symptoms: {', '.join(symptoms or [])}"
        vector = self.vectorizer.transform([query]).astype("float32").toarray()
        faiss.normalize_L2(vector)
        k = min(top_k, len(self.records))
        scores, indices = self.index.search(vector, k)
        return [
            {**self.records[i], "similarity": float(score)}
            for score, i in zip(scores[0], indices[0]) if i >= 0
        ]
