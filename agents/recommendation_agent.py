import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    api_key=os.getenv("NVIDIA_API_KEY"),
)
MODEL = os.getenv("NVIDIA_REASONING_MODEL", "nvidia/nemotron-3-super-120b-a12b")

def generate_recommendations(diagnosis, evidence):
    prompt = f"""You are an agricultural recommendation agent.
Diagnosis:
{json.dumps(diagnosis, indent=2)}
PlantVillage evidence:
{json.dumps(evidence, indent=2)}

Return ONLY valid JSON:
{{
  "organic_recommendations": [],
  "prevention": [],
  "warning": ""
}}
Rules:
- Treat diagnosis as an initial AI assessment, not certainty.
- Prefer cultural/non-chemical organic practices.
- Do not invent pesticide doses, product names, or chemical mixtures.
- If a treatment depends on local law, crop, region, or organic certification, say so.
- Keep advice practical and concise."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        top_p=0.95,
        max_tokens=2500,
        extra_body={"chat_template_kwargs": {"enable_thinking": True, "low_effort": True}},
    )
    return (response.choices[0].message.content or "").strip()
