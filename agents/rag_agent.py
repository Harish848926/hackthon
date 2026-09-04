from rag.retriever import PlantVillageRetriever

_retriever = None

def retrieve_plantvillage_evidence(diagnosis):
    global _retriever
    if _retriever is None:
        _retriever = PlantVillageRetriever()
    return _retriever.search(
        crop=diagnosis.get("plant", "Unknown"),
        disease=diagnosis.get("disease", "Unknown"),
        symptoms=diagnosis.get("symptoms", []),
        top_k=5,
    )
