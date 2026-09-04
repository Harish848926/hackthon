from agents.vision_agent import analyze_plant_image
from agents.rag_agent import retrieve_plantvillage_evidence
from agents.recommendation_agent import generate_recommendations

def run_plant_analysis(image_bytes, mime_type):
    diagnosis = analyze_plant_image(image_bytes, mime_type)
    evidence = retrieve_plantvillage_evidence(diagnosis)
    recommendations = generate_recommendations(diagnosis, evidence)
    return {"diagnosis": diagnosis, "evidence": evidence, "recommendations": recommendations}
