# Breast Cancer Survival Prediction RAG Assistant (SEER-Based)

An educational decision-support tool that takes a patient's clinical information and passes it to **a model you trained yourself** to predict survival status (Alive / Dead). It then uses RAG and a language model to explain why this result was produced and which risk factors appear in the patient's profile.

## ⚠️ This Is Not a Diagnostic Tool

The output is only a statistical estimate based on historical SEER data patterns. This tool does not diagnose disease, does not replace an oncologist's judgment, and should not be the sole basis for any treatment decision. Explanations are generated only from the reference texts in `knowledge_base.json`; the tool must not make additional claims beyond those sources.

## Overview

1. The patient is described using 14 clinical features (age, tumor stage, hormone status, etc.).
2. Text values (such as `T2`) are converted to numbers and passed to `classifier.joblib`.
3. The model returns `Alive` or `Dead`, along with a confidence percentage.
4. Semantic search (`sentence-transformers` and FAISS) retrieves the most relevant concepts from the knowledge base.
5. The language model generates an explanation using only those concepts and the patient's profile.

Unlike the Wisconsin project, which determines whether a mass is benign or malignant, this project estimates the **prognosis** of a patient who has already been diagnosed.

## Project Structure

```text
medical_rag_assistant/
├── medical_rag_assistant.py   # Main file; run this
├── knowledge_base.json        # 15 SEER concepts for retrieval
├── classifier.joblib          # Your model (you must provide this file)
├── requirements.txt
├── .env                       # API key and settings (never commit this file)
├── .gitignore
├── README.md                  # This file
└── README.en.md               # English version
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

Place your trained model file named `classifier.joblib` next to the script (or specify another path using `CLASSIFIER_PATH`). Then fill in your `.env` file:

```env
LLM_API_KEY=your-api-key
LLM_MODEL=gpt-4o-mini
# LLM_BASE_URL=
CLASSIFIER_PATH=classifier.joblib
```

### Using OpenRouter

To use other models (for example, Qwen or GLM through OpenRouter), change these two values:

```env
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=model-id-from-the-OpenRouter-list
```

If you leave `LLM_API_KEY` empty, the program will not crash. Instead of a language-generated explanation, it will display the retrieved reference texts.

## Run

```bash
python medical_rag_assistant.py
```

The program prompts for each feature one by one, rejects invalid values and asks again, then prints the prediction and explanation.

## Model File Format

`classifier.joblib` can be either of the following:

1. **A dictionary** (recommended):

   ```python
   joblib.dump({"model": clf,
                "feature_names": [...],
                "target_names": ["Alive", "Dead"]}, "classifier.joblib")
   ```

2. **The model object itself**. In this case, the default list of 14 features below is used (if they differ, update `DEFAULT_FEATURE_NAMES` and `DEFAULT_TARGET_NAMES` at the top of the script).

## Features and Allowed Values

| Feature | Values |
|---|---|
| `Age` | Numeric |
| `Race` | White, Black, Other |
| `Marital Status` | Single, Married, Separated, Divorced, Widowed |
| `T Stage` | T1, T2, T3, T4 |
| `N Stage` | N1, N2, N3 |
| `6th Stage` | IIA, IIB, IIIA, IIIB, IIIC |
| `differentiate` | Well differentiated, Moderately differentiated, Poorly differentiated, Undifferentiated |
| `Grade` | 1, 2, 3, 4 |
| `A Stage` | Regional, Distant |
| `Tumor Size` | Numeric (millimeters) |
| `Estrogen Status` | Positive, Negative |
| `Progesterone Status` | Positive, Negative |
| `Regional Node Examined` | Numeric |
| `Reginol Node Positive` | Numeric |

## ⚠️ Critical: Encoding Must Match the Model's Training

The numeric order assigned to text values in `ENCODING_MAPS` (for example, `White=0, Black=1, Other=2`) **must exactly match the order used when training the model**. If you used `LabelEncoder`, check `encoder.classes_` and align `ENCODING_MAPS` accordingly. If they do not match, the program will produce predictions without raising an error, but those predictions will be meaningless. This is the most dangerous part of the project and must be checked before relying on the output.

## scikit-learn Version Compatibility Error

If you trained the model in one environment (such as Kaggle) and load it in another (such as Colab or a server), you may encounter an error like `node array from the pickle has an incompatible dtype`. This happens because the internal structure of scikit-learn trees changes between versions. The solution is to install the same scikit-learn version that was used to train the model:

```bash
pip install scikit-learn==training-version
```

Get the version from the training notebook using:

```python
import sklearn
print(sklearn.__version__)
```

## Preventing Data Leakage

The `Survival Months` column is intentionally excluded from the inputs and prompt. This value is only known after the outcome has occurred, so using it in a prediction model would constitute data leakage.

## Ethical Considerations

Variables such as race and marital status are statistically associated with survival differences in SEER data, but these differences mostly reflect access to treatment and social factors rather than biological causes. Any result that relies partly on these variables should be interpreted with caution.

## Limitations

- The knowledge base contains only 15 concepts and is not a comprehensive oncology reference.
- Model accuracy depends on your own training and evaluation. This project does not claim any particular accuracy score, so if you include it in a report or résumé, provide your own evaluation results using a separate test set.
- The SEER data used relates to patients from 2006 to 2010 and two specific types of invasive breast cancer; it cannot be generalized to all types and populations.
- Conversations and predictions are not stored, and there is no authentication.

## Security

Never put your API key directly in code or notebooks, and never commit `.env`. If a key is exposed anywhere, revoke it immediately and generate a new one.
