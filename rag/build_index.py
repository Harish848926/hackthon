import os
import pickle
import faiss

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# CONFIG
# ============================================================

DATASET_NAME = "geraldmc/plantvillage-full"

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

INDEX_DIR = os.path.join(
    BASE_DIR,
    "data",
    "index"
)

os.makedirs(
    INDEX_DIR,
    exist_ok=True
)


print("=" * 70)
print("🌱 PLANT DISEASE AI")
print("PlantVillage RAG Builder")
print("=" * 70)


# ============================================================
# LOAD DATASET
# ============================================================

print("\n📥 Loading PlantVillage...")

dataset = load_dataset(
    DATASET_NAME,
    split="train"
)


print("\n✅ Dataset loaded")

print(
    "Rows:",
    len(dataset)
)

print(
    "Columns:",
    dataset.column_names
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "host",
    "disease",
    "class_label"
]


for column in required_columns:

    if column not in dataset.column_names:

        raise RuntimeError(
            f"Missing column: {column}\n"
            f"Available columns: "
            f"{dataset.column_names}"
        )


print(
    "\n✅ Required columns found:"
)

print(
    "   host"
)

print(
    "   disease"
)

print(
    "   class_label"
)


# ============================================================
# CREATE RECORDS
# ============================================================

records = []


print(
    "\n🔄 Creating RAG knowledge..."
)


for row in dataset:

    host = row["host"]

    disease = row["disease"]

    class_label = row["class_label"]


    # Convert to strings

    host = str(
        host
    ).strip()


    disease = str(
        disease
    ).strip()


    class_label = str(
        class_label
    ).strip()


    # Clean underscores

    host = host.replace(
        "_",
        " "
    )


    disease = disease.replace(
        "_",
        " "
    )


    class_label = class_label.replace(
        "_",
        " "
    )


    records.append({

        "crop": host,

        "disease": disease,

        "class_label": class_label,

        "source": "PlantVillage"

    })


# ============================================================
# REMOVE DUPLICATES
# ============================================================

unique_records = {}


for record in records:

    key = (

        record["crop"],

        record["disease"]

    )


    unique_records[key] = record


records = list(
    unique_records.values()
)


print(
    "\n📚 Unique disease classes:",
    len(records)
)


# ============================================================
# CREATE DOCUMENTS
# ============================================================

texts = []


for record in records:

    text = (

        "Plant: "
        + record["crop"]
        + ". "

        "Disease: "
        + record["disease"]
        + ". "

        "PlantVillage class: "
        + record["class_label"]
        + ". "

        "Source: PlantVillage."

    )


    texts.append(
        text
    )


# ============================================================
# TF-IDF
# ============================================================

print(
    "\n🔎 Creating TF-IDF vectors..."
)


vectorizer = TfidfVectorizer(

    lowercase=True,

    ngram_range=(1, 2),

    min_df=1

)


matrix = vectorizer.fit_transform(
    texts
)


matrix = matrix.astype(
    "float32"
).toarray()


# ============================================================
# NORMALIZE
# ============================================================

faiss.normalize_L2(
    matrix
)


# ============================================================
# CREATE FAISS CPU INDEX
# ============================================================

print(
    "🧠 Creating FAISS CPU index..."
)


dimension = matrix.shape[1]


index = faiss.IndexFlatIP(
    dimension
)


index.add(
    matrix
)


# ============================================================
# SAVE FILES
# ============================================================

index_file = os.path.join(

    INDEX_DIR,

    "plantvillage.faiss"

)


records_file = os.path.join(

    INDEX_DIR,

    "records.pkl"

)


vectorizer_file = os.path.join(

    INDEX_DIR,

    "vectorizer.pkl"

)


faiss.write_index(

    index,

    index_file

)


with open(

    records_file,

    "wb"

) as f:

    pickle.dump(

        records,

        f

    )


with open(

    vectorizer_file,

    "wb"

) as f:

    pickle.dump(

        vectorizer,

        f

    )


# ============================================================
# SHOW SAMPLE
# ============================================================

print(
    "\n📋 Sample RAG records:"
)


for record in records[:10]:

    print(

        f"   🌿 {record['crop']} "
        f"→ {record['disease']}"

    )


# ============================================================
# COMPLETE
# ============================================================

print("\n")

print(
    "=" * 70
)

print(
    "🎉 PLANTVILLAGE RAG CREATED!"
)

print(
    "=" * 70
)


print(
    "\nFAISS index:"
)

print(
    index_file
)


print(
    "\nRecords:"
)

print(
    records_file
)


print(
    "\nVectorizer:"
)

print(
    vectorizer_file
)


print(
    "\nTotal records:",
    len(records)
)


print(
    "\nNext command:"
)

print(
    "streamlit run app.py"
)