# utils.py
import pandas as pd
import numpy as np
import re
import torch
from sentence_transformers import SentenceTransformer, util
from razdel import tokenize
from nltk.corpus import stopwords
import pymorphy2
import nltk
nltk.download('stopwords')

model = SentenceTransformer("cointegrated/rubert-tiny2")
morph = pymorphy2.MorphAnalyzer()
stop_words = set(stopwords.words("russian"))

# Глобальный синонимичный маппинг для базовых слов
SYNONYM_MAP = {
    "симкарта": ["сим карта", "сим-карта", "симка", "симки", "сим"],
    "карта": ["карточка", "карточку"],
    "банк": ["банковский", "банка"],
    "утеряна": ["потеряна", "пропала"]
}


def normalize(text):
    tokens = [_.text.lower() for _ in tokenize(text)]
    lemmas = [morph.parse(t)[0].normal_form for t in tokens if t not in stop_words and t.isalpha()]
    return " ".join(lemmas)


def synonym_replace(text):
    norm_text = normalize(text)
    for base, syns in SYNONYM_MAP.items():
        for word in syns:
            pattern = re.compile(rf"\b{word}\b")
            norm_text = pattern.sub(base, norm_text)
    return norm_text


def load_data(file):
    df = pd.read_excel(file)
    df = df.dropna(subset=["text"])
    df["norm_text"] = df["text"].apply(synonym_replace)
    embeddings = model.encode(df["norm_text"].tolist(), convert_to_tensor=True, show_progress_bar=False)
    return df, embeddings


def semantic_search(query, df, embeddings, top_k=5):
    norm_query = synonym_replace(query)
    query_emb = model.encode(norm_query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_emb, embeddings)[0]
    top_results = torch.topk(scores, k=top_k)

    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        results.append({
            "text": df.iloc[idx]["text"],
            "label": df.iloc[idx]["label"],
            "code": df.iloc[idx]["code"],
            "score": score.item()
        })

    # Ключевое совпадение для коротких слов
    keyword_matches = []
    words = set(query.lower().split())
    short_words = [w for w in words if len(w) <= 5]

    for i, row in df.iterrows():
        for w in short_words:
            if re.search(rf"\b{w}\b", row["text"].lower()):
                keyword_matches.append({
                    "text": row["text"],
                    "label": row["label"],
                    "code": row["code"]
                })
                break

    return results, keyword_matches
