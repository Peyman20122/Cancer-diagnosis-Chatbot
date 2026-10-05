from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import faiss
import joblib
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_PATH = BASE_DIR / "knowledge_base.json"
CLASSIFIER_PATH = Path(os.environ.get("CLASSIFIER_PATH", BASE_DIR / "classifier.joblib"))
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

LLM_API_KEY = os.environ.get("LLM_API_KEY")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL")  # leave unset for the real OpenAI endpoint
USE_LLM = bool(LLM_API_KEY)

DEFAULT_FEATURE_NAMES = [
    "Age", "Race", "Marital Status", "T Stage", "N Stage", "6th Stage",
    "differentiate", "Grade", "A Stage", "Tumor Size",
    "Estrogen Status", "Progesterone Status",
    "Regional Node Examined", "Reginol Node Positive",
]
DEFAULT_TARGET_NAMES = ["Alive", "Dead"]

ENCODING_MAPS = {
    "Race": {"White": 0, "Black": 1, "Other": 2},
    "Marital Status": {"Single": 0, "Married": 1, "Separated": 2, "Divorced": 3, "Widowed": 4},
    "T Stage": {"T1": 0, "T2": 1, "T3": 2, "T4": 3},
    "N Stage": {"N1": 0, "N2": 1, "N3": 2},
    "6th Stage": {"IIA": 0, "IIB": 1, "IIIA": 2, "IIIB": 3, "IIIC": 4},
    "differentiate": {
        "Well differentiated": 0,
        "Moderately differentiated": 1,
        "Poorly differentiated": 2,
        "Undifferentiated": 3,
    },
    "Grade": {"1": 0, "2": 1, "3": 2, "4": 3},
    "A Stage": {"Regional": 0, "Distant": 1},
    "Estrogen Status": {"Positive": 0, "Negative": 1},
    "Progesterone Status": {"Positive": 0, "Negative": 1},
}
NUMERIC_FEATURES = {"Age", "Tumor Size", "Regional Node Examined", "Reginol Node Positive"}


def encode_feature(name: str, raw_value: str):
    name = name.strip()
    if name in NUMERIC_FEATURES:
        try:
            return float(raw_value)
        except (ValueError, TypeError):
            raise ValueError(f"Feature '{name}' expects a numeric value, got: {raw_value!r}")
    mapping = ENCODING_MAPS.get(name, {})
    if not mapping:
        raise ValueError(f"Unknown feature: '{name}'")
    if raw_value not in mapping:
        raise ValueError(f"Value '{raw_value}' for '{name}' not recognized. Valid options: {list(mapping.keys())}")
    return mapping[raw_value]


def load_classifier():
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"No classifier found at {CLASSIFIER_PATH}. Place your trained "
            f"classifier.joblib there, or set CLASSIFIER_PATH in .env."
        )
    loaded = joblib.load(CLASSIFIER_PATH)
    if isinstance(loaded, dict) and "model" in loaded:
        return (
            loaded["model"],
            loaded.get("feature_names", DEFAULT_FEATURE_NAMES),
            loaded.get("target_names", DEFAULT_TARGET_NAMES),
        )
    return loaded, DEFAULT_FEATURE_NAMES, DEFAULT_TARGET_NAMES


def load_knowledge_base() -> list[dict]:
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class MedicalRAGChatbot:
    def __init__(self, clf, feature_names: list[str], target_names: list[str],
                 embedder: SentenceTransformer, index, kb: list[dict]) -> None:
        self.clf = clf
        self.feature_names = feature_names
        self.target_names = target_names
        self.embedder = embedder
        self.index = index
        self.kb = kb
        self.client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL) if USE_LLM else None

    def build_system_prompt(self, patient_values: dict, prediction: str, confidence: float) -> str:
        # Survival Months is known only retrospectively and isn't collected as
        # an input here, so there's nothing to exclude, but the guard stays
        # in case a caller passes it in anyway.
        exclude = {"Survival Months"}
        lines = []
        for fname in self.feature_names:
            if fname in exclude:
                continue
            raw_val = patient_values.get(fname)
            if raw_val is None:
                continue
            label_map = ENCODING_MAPS.get(fname)
            display = label_map and {v: k for k, v in label_map.items()}.get(int(raw_val))
            lines.append(f"  - {fname}: {display if display else raw_val}")
        profile_block = "\n".join(lines)

        return (
            "You are a Clinical Decision Support (CDS) assistant specializing in breast "
            "cancer survival analysis. Your role is to help clinicians interpret ML-based "
            "survival predictions and provide evidence-based context -- NOT to replace "
            "clinical judgment.\n\n"
            f"## Patient Profile\n{profile_block}\n\n"
            f"## ML Model Output\n  - Predicted Status: {prediction}\n"
            f"  - Model Confidence: {confidence:.1%}\n\n"
            "Guidelines:\n"
            "- Use ONLY the provided context passages to support your explanation.\n"
            "- Translate encoded values to clinically meaningful language.\n"
            "- Highlight risk factors present in this patient's profile.\n"
            "- Always recommend consultation with a qualified oncologist."
        )

    def predict(self, encoded_values: list[float]) -> dict:
        x = np.array([encoded_values])
        pred_idx = self.clf.predict(x)[0]
        proba = self.clf.predict_proba(x)[0]
        return {"prediction": self.target_names[pred_idx], "confidence": float(proba[pred_idx])}

    def retrieve(self, query: str, k: int = 4) -> list[dict]:
        query_vec = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, indices = self.index.search(query_vec.astype(np.float32), k)
        return [self.kb[idx] for idx in indices[0] if idx != -1]

    def explain(self, patient_values: dict, prediction: str, confidence: float,
                passages: list[dict], question: str) -> str:
        context = "\n\n".join(f"[{p['topic']}]\n{p['text']}" for p in passages)

        if not self.client:
            lines = [f"Prediction: {prediction.upper()} (confidence: {confidence:.2%})", ""]
            lines += [f"- {p['text']}" for p in passages]
            lines.append("\nNote: set LLM_API_KEY in .env for a natural-language explanation.")
            return "\n".join(lines)

        system_prompt = self.build_system_prompt(patient_values, prediction, confidence)
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    def ask(self, patient_values: dict, question: str) -> str:
        encoded = [patient_values[f] for f in self.feature_names]
        result = self.predict(encoded)
        passages = self.retrieve(f"{question} Prediction: {result['prediction']}.")
        return self.explain(patient_values, result["prediction"], result["confidence"], passages, question)


def build_chatbot() -> MedicalRAGChatbot:
    kb = load_knowledge_base()
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [entry["text"] for entry in kb]
    vectors = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype(np.float32))

    clf, feature_names, target_names = load_classifier()
    return MedicalRAGChatbot(clf, feature_names, target_names, embedder, index, kb)


def main() -> None:
    bot = build_chatbot()

    print("TUMBot: Welcome! I am TUMBot, your Clinical Decision Support assistant.")
    print("TUMBot: Please enter the following patient details:\n")

    patient_values: dict[str, float] = {}
    for fname in bot.feature_names:
        while True:
            raw = input(f"  {fname}: ").strip()
            try:
                patient_values[fname] = encode_feature(fname, raw)
                break
            except ValueError as e:
                print(f"  [error] {e}. Please try again.")

    print()
    encoded = [patient_values[f] for f in bot.feature_names]
    result = bot.predict(encoded)
    print(f"TUMBot: Model prediction: {result['prediction'].upper()} "
          f"({result['confidence']:.1%} confidence). Here's the reasoning:\n")

    answer = bot.ask(patient_values, "Why was this prediction made and what are the key risk factors?")
    print(answer)
    print(
        "\nTUMBot: Reminder -- this tool supports clinical decision-making, it does not "
        "replace it. Final diagnosis and treatment should always involve a qualified oncologist."
    )


if __name__ == "__main__":
    main()
