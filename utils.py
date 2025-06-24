import pandas as pd
import requests
import re
import json
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
import pymorphy2

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
morph = pymorphy2.MorphAnalyzer()

GITHUB_EXCEL_URLS = [
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data1.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data2.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data3.xlsx"
]

# Загрузка синонимов
try:
    with open("synonyms.json", "r", encoding="utf-8") as f:
        SYNONYM_GROUPS = json.load(f)
except Exception as e:
    print(f"Ошибка загрузки синонимов: {e}")
    SYNONYM_GROUPS = []

def lemmatize(word):
    return morph.parse(word)[0].normal_form.lower()

def build_synonym_map(groups):
    mapping = {}
    for group in groups:
        norm_forms = {lemmatize(w) for w in group}
        for form in norm_forms:
            for synonym in norm_forms:
                mapping[form] = synonym
    return mapping

SYNONYM_MAP = build_synonym_map(SYNONYM_GROUPS)

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"[-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    norm_tokens = [SYNONYM_MAP.get(lemmatize(t), lemmatize(t)) for t in tokens]
    return " ".join(norm_tokens)

def load_excel(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Ошибка загрузки файла: {url}")
    df = pd.read_excel(BytesIO(response.content))

    topic_cols = [col for col in df.columns if col.lower().startswith("topics")]
    if not topic_cols:
        raise KeyError("Не найдены колонки с темами")

    df['topics'] = df[topic_cols].fillna('').agg(lambda x: [i for i in x.tolist() if i], axis=1)
    df['phrase_proc'] = df['phrase'].apply(preprocess)
    return df[['phrase', 'phrase_proc', 'topics']]

def load_all_excels():
    dfs = []
    for url in GITHUB_EXCEL_URLS:
        try:
            df = load_excel(url)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Проблема с файлом {url}: {e}")
    if not dfs:
        raise RuntimeError("Не удалось загрузить ни один Excel-файл.")
    return pd.concat(dfs, ignore_index=True)

def semantic_search(query, df, top_k=5, threshold=0.5):
    query_proc = preprocess(query)
    query_emb = model.encode(query_proc, convert_to_tensor=True)
    phrase_embs = model.encode(df['phrase_proc'].tolist(), convert_to_tensor=True)
    sims = util.pytorch_cos_sim(query_emb, phrase_embs)[0]

    results = []
    for idx, score in enumerate(sims):
        score = float(score)
        if score >= threshold:
            phrase = df.iloc[idx]['phrase']
            topics = df.iloc[idx]['topics']
            results.append((score, phrase, topics))

    # Расширенный точный поиск по леммам и синонимам
    if len(query.strip()) <= 5:
        query_lemma = lemmatize(query.strip())
        query_syns = {query_lemma, SYNONYM_MAP.get(query_lemma, query_lemma)}
        matches = df[df['phrase_proc'].apply(lambda p: any(f" {syn} " in f" {p} " for syn in query_syns))]
        for _, row in matches.iterrows():
            phrase, topics = row['phrase'], row['topics']
            if (1.0, phrase, topics) not in results:
                results.append((1.0, phrase, topics))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]
