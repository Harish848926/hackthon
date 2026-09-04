import os
import base64
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    api_key=os.getenv("NVIDIA_API_KEY"),
)
VISION_MODEL = os.getenv("NVIDIA_VISION_MODEL", "meta/llama-3.2-90b-vision-instruct")

def _data_url(image_bytes: bytes, mime_type: str) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode()}"

def analyze_plant_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
    if not os.getenv("NVIDIA_API_KEY"):
        raise RuntimeError("NVIDIA_API_KEY is missing. Copy .env.example to .env and add your key.")

    prompt = """You are a plant-disease vision specialist.
Analyze the uploaded plant/leaf image. Return ONLY valid JSON:
{
  "plant": "",
  "disease": "",
  "confidence": 0.0,
  "symptoms": [],
  "visual_evidence": [],
  "alternative_diseases": []
}
Rules: confidence is 0..1; describe only visible evidence; if unclear use Unknown; do not give treatments."""
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _data_url(image_bytes, mime_type)}}
        ]}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1200,
    )
    text = (response.choices[0].message.content or "").strip()
    if "```" in text:
        text = text.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = {
            "plant": "Unknown",
            "disease": "Unable to parse diagnosis",
            "confidence": 0.0,
            "symptoms": [],
            "visual_evidence": [text],
            "alternative_diseases": [],
        }
    data["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0) or 0)))
    return data
