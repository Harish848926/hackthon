import streamlit as st
import json
import re

from agents.orchestrator import run_plant_analysis


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Plant Disease AI",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🌱 Plant Disease AI")
st.caption(
    "GPU-Free Multi-Agent Plant Disease Diagnosis "
    "using NVIDIA AI + PlantVillage RAG"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(value):
    """
    Convert any value into clean display text.
    """

    if value is None:
        return ""

    return str(value).strip()


def parse_json_from_text(text):
    """
    Extract JSON object from NVIDIA/model response.
    """

    if not text:
        return None

    text = str(text).strip()

    # Remove markdown code fences
    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = text.replace(
        "```",
        ""
    ).strip()

    # Try entire response
    try:
        return json.loads(text)
    except Exception:
        pass

    # Find JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:

        json_text = text[
            start:end + 1
        ]

        try:
            return json.loads(json_text)
        except Exception:
            pass

    return None


def parse_text_diagnosis(text):
    """
    Parse responses such as:

    Plant: Tomato
    Disease: Septoria Leaf Spot
    Confidence: 0.8
    """

    if not text:
        return {}

    text = str(text)

    result = {}

    # Plant
    match = re.search(
        r"(?:Plant|Crop)\s*:\s*(.+)",
        text,
        re.IGNORECASE
    )

    if match:
        result["plant"] = (
            match.group(1)
            .strip()
            .split("\n")[0]
        )

    # Disease
    match = re.search(
        r"(?:Disease|Diagnosis)\s*:\s*(.+)",
        text,
        re.IGNORECASE
    )

    if match:
        result["disease"] = (
            match.group(1)
            .strip()
            .split("\n")[0]
        )

    # Confidence
    match = re.search(
        r"Confidence\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        text,
        re.IGNORECASE
    )

    if match:

        try:

            confidence = float(
                match.group(1)
            )

            if confidence > 1:
                confidence /= 100

            result["confidence"] = confidence

        except Exception:
            result["confidence"] = 0.0

    return result


def normalize_diagnosis(result):
    """
    Extract diagnosis regardless of whether the
    orchestrator returns a dictionary or text.
    """

    if not result:
        return {
            "plant": "Unknown",
            "disease": "Unknown",
            "confidence": 0.0,
            "symptoms": [],
            "visual_evidence": [],
            "alternative_diseases": []
        }

    # ========================================================
    # CASE 1
    # result["diagnosis"] is already a dictionary
    # ========================================================

    if isinstance(result, dict):

        diagnosis = result.get(
            "diagnosis"
        )

        if isinstance(
            diagnosis,
            dict
        ):

            data = diagnosis

        else:

            # =================================================
            # CASE 2
            # result itself contains diagnosis fields
            # =================================================

            if (
                "plant" in result
                or "disease" in result
                or "confidence" in result
            ):

                data = result

            else:

                # =================================================
                # CASE 3
                # result["answer"] contains model output
                # =================================================

                answer = result.get(
                    "answer"
                )

                if answer:

                    parsed = parse_json_from_text(
                        answer
                    )

                    if parsed:
                        data = parsed
                    else:
                        data = parse_text_diagnosis(
                            answer
                        )

                else:

                    data = {}

    else:

        # ========================================================
        # CASE 4
        # result is directly text
        # ========================================================

        parsed = parse_json_from_text(
            result
        )

        if parsed:
            data = parsed
        else:
            data = parse_text_diagnosis(
                result
            )

    # ========================================================
    # Normalize values
    # ========================================================

    plant = data.get(
        "plant",
        data.get(
            "crop",
            "Unknown"
        )
    )

    disease = data.get(
        "disease",
        data.get(
            "diagnosis",
            "Unknown"
        )
    )

    confidence = data.get(
        "confidence",
        0.0
    )

    # Convert confidence
    try:

        confidence = float(
            confidence
        )

    except Exception:

        confidence = 0.0

    if confidence > 1:

        confidence /= 100

    confidence = max(
        0.0,
        min(
            1.0,
            confidence
        )
    )

    # ========================================================
    # Symptoms
    # ========================================================

    symptoms = data.get(
        "symptoms",
        []
    )

    if isinstance(
        symptoms,
        str
    ):

        symptoms = [
            symptoms
        ]

    # ========================================================
    # Visual evidence
    # ========================================================

    visual_evidence = data.get(
        "visual_evidence",
        []
    )

    if isinstance(
        visual_evidence,
        str
    ):

        visual_evidence = [
            visual_evidence
        ]

    # ========================================================
    # Alternative diseases
    # ========================================================

    alternatives = data.get(
        "alternative_diseases",
        []
    )

    if isinstance(
        alternatives,
        str
    ):

        alternatives = [
            alternatives
        ]

    return {

        "plant":
            clean_text(plant)
            or "Unknown",

        "disease":
            clean_text(disease)
            or "Unknown",

        "confidence":
            confidence,

        "symptoms":
            symptoms,

        "visual_evidence":
            visual_evidence,

        "alternative_diseases":
            alternatives
    }


def get_list(value):
    """
    Safely convert result into a list.
    """

    if value is None:
        return []

    if isinstance(
        value,
        list
    ):
        return value

    if isinstance(
        value,
        tuple
    ):
        return list(value)

    if isinstance(
        value,
        str
    ):
        return [
            value
        ]

    return [value]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System")

    st.markdown(
        """
        **Vision**
        
        NVIDIA Vision NIM
        
        **RAG**
        
        PlantVillage
        
        **Reasoning**
        
        NVIDIA Nemotron
        
        **Runtime**
        
        GPU-Free / API based
        """
    )

    st.divider()

    st.info(
        "Upload a clear plant leaf image "
        "for better diagnosis."
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.header("📷 Upload Plant Image")

uploaded = st.file_uploader(
    "Choose a plant leaf image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# ============================================================
# IMAGE PREVIEW
# ============================================================

if uploaded:

    col1, col2 = st.columns(
        [1, 2]
    )

    with col1:

        st.image(
            uploaded,
            caption="Uploaded Plant Image",
            use_container_width=True
        )

    with col2:

        st.markdown(
            "### 🔬 Ready for Analysis"
        )

        st.write(
            f"**File:** {uploaded.name}"
        )

        st.write(
            f"**Type:** {uploaded.type}"
        )

        st.write(
            f"**Size:** "
            f"{uploaded.size / 1024:.1f} KB"
        )


# ============================================================
# ANALYZE BUTTON
# ============================================================

if uploaded:

    st.divider()

    analyze_button = st.button(
        "🔍 Analyze Plant",
        type="primary",
        use_container_width=True
    )

    if analyze_button:

        try:

            # ====================================================
            # RUN MULTI-AGENT SYSTEM
            # ====================================================

            with st.spinner(
                "🤖 AI agents analyzing plant image..."
            ):

                result = run_plant_analysis(
                    uploaded.getvalue(),
                    uploaded.type
                )


            # ====================================================
            # DEBUG
            # ====================================================

            if not result:

                st.error(
                    "No result returned from AI system."
                )

                st.stop()


            # ====================================================
            # EXTRACT DIAGNOSIS
            # ====================================================

            diagnosis = normalize_diagnosis(
                result
            )

            plant = diagnosis[
                "plant"
            ]

            disease = diagnosis[
                "disease"
            ]

            confidence = diagnosis[
                "confidence"
            ]

            symptoms = diagnosis[
                "symptoms"
            ]

            visual_evidence = diagnosis[
                "visual_evidence"
            ]

            alternatives = diagnosis[
                "alternative_diseases"
            ]


            # ====================================================
            # STORE IN SESSION
            # ====================================================

            st.session_state[
                "diagnosis"
            ] = diagnosis

            st.session_state[
                "analysis_result"
            ] = result


            # ====================================================
            # DIAGNOSIS RESULT
            # ====================================================

            st.header(
                "🩺 Disease Diagnosis"
            )

            col1, col2, col3 = st.columns(
                3
            )


            # ====================================================
            # PLANT
            # ====================================================

            with col1:

                st.markdown(
                    "### 🌱 Plant"
                )

                st.success(
                    plant
                )


            # ====================================================
            # DISEASE
            # ====================================================

            with col2:

                st.markdown(
                    "### 🦠 Disease"
                )

                if disease.lower() == "unknown":

                    st.warning(
                        disease
                    )

                else:

                    st.error(
                        disease
                    )


            # ====================================================
            # CONFIDENCE
            # ====================================================

            with col3:

                st.markdown(
                    "### 🎯 Model Confidence"
                )

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.1f}%"
                )

                st.progress(
                    confidence
                )


            # ====================================================
            # SYMPTOMS
            # ====================================================

            st.divider()

            st.subheader(
                "🔎 Symptoms"
            )

            if symptoms:

                for symptom in get_list(
                    symptoms
                ):

                    st.markdown(
                        f"• {symptom}"
                    )

            else:

                st.info(
                    "No symptoms identified."
                )


            # ====================================================
            # VISUAL EVIDENCE
            # ====================================================

            st.subheader(
                "👁️ Visual Evidence"
            )

            if visual_evidence:

                for evidence in get_list(
                    visual_evidence
                ):

                    st.markdown(
                        f"• {evidence}"
                    )

            else:

                st.info(
                    "No visual evidence available."
                )


            # ====================================================
            # ALTERNATIVE DISEASES
            # ====================================================

            if alternatives:

                st.subheader(
                    "🔄 Alternative Diseases"
                )

                for alternative in get_list(
                    alternatives
                ):

                    st.markdown(
                        f"• {alternative}"
                    )


            # ====================================================
            # PLANTVILLAGE RAG
            # ====================================================

            st.divider()

            st.header(
                "📚 PlantVillage RAG Evidence"
            )

            evidence = []

            if isinstance(
                result,
                dict
            ):

                evidence = result.get(
                    "evidence",
                    result.get(
                        "rag_evidence",
                        []
                    )
                )

            evidence = get_list(
                evidence
            )

            if evidence:

                for item in evidence:

                    if isinstance(
                        item,
                        dict
                    ):

                        label = item.get(
                            "label",
                            item.get(
                                "class_label",
                                item.get(
                                    "disease",
                                    "PlantVillage result"
                                )
                            )
                        )

                        similarity = item.get(
                            "similarity",
                            item.get(
                                "score",
                                None
                            )
                        )

                        if similarity is not None:

                            try:

                                similarity_text = (
                                    f"{float(similarity):.3f}"
                                )

                            except Exception:

                                similarity_text = str(
                                    similarity
                                )

                            st.markdown(
                                f"**{label}**  "
                                f"· similarity {similarity_text}"
                            )

                        else:

                            st.markdown(
                                f"**{label}**"
                            )

                    else:

                        st.markdown(
                            f"• {item}"
                        )

            else:

                st.info(
                    "No PlantVillage evidence returned."
                )


            # ====================================================
            # ORGANIC RECOMMENDATIONS
            # ====================================================

            st.divider()

            st.header(
                "🌱 Organic Recommendations"
            )

            recommendations = []

            if isinstance(
                result,
                dict
            ):

                recommendations = result.get(
                    "recommendations",
                    result.get(
                        "organic_recommendations",
                        []
                    )
                )

            recommendations = get_list(
                recommendations
            )

            if recommendations:

                for recommendation in recommendations:

                    if isinstance(
                        recommendation,
                        dict
                    ):

                        text = recommendation.get(
                            "recommendation",
                            recommendation.get(
                                "text",
                                str(recommendation)
                            )
                        )

                    else:

                        text = str(
                            recommendation
                        )

                    st.markdown(
                        f"✅ {text}"
                    )

            else:

                st.info(
                    "No organic recommendations returned."
                )


            # ====================================================
            # PREVENTION
            # ====================================================

            st.header(
                "🛡️ Prevention"
            )

            prevention = []

            if isinstance(
                result,
                dict
            ):

                prevention = result.get(
                    "prevention",
                    []
                )

            prevention = get_list(
                prevention
            )

            if prevention:

                for item in prevention:

                    if isinstance(
                        item,
                        dict
                    ):

                        text = item.get(
                            "text",
                            item.get(
                                "recommendation",
                                str(item)
                            )
                        )

                    else:

                        text = str(
                            item
                        )

                    st.markdown(
                        f"• {text}"
                    )

            else:

                st.info(
                    "No prevention information returned."
                )


            # ====================================================
            # SAFETY NOTE
            # ====================================================

            st.divider()

            st.warning(
                "AI diagnosis is an initial assessment. "
                "Verify important crop-management decisions "
                "with a qualified agricultural expert. "
                "Check that any treatment is compatible "
                "with your organic certification and local "
                "regulations."
            )


        # ========================================================
        # ERROR HANDLING
        # ========================================================

        except Exception as e:

            st.error(
                "❌ Analysis failed"
            )

            st.exception(
                e
            )


# ============================================================
# INITIAL SCREEN
# ============================================================

else:

    st.info(
        "👆 Upload a plant leaf image to start diagnosis."
    )

    st.markdown(
        """
        ### 🚀 MVP Pipeline

        **1. Image Upload**  
        Upload JPG, PNG or WEBP.

        **2. NVIDIA Vision**  
        Analyze the leaf image.

        **3. PlantVillage RAG**  
        Retrieve relevant disease evidence.

        **4. NVIDIA Nemotron**  
        Generate disease explanation and
        organic recommendations.

        **5. Streamlit Result**  
        Display diagnosis, confidence,
        symptoms, evidence and recommendations.
        """
    )