import pandas as pd
import requests
import re
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
from nltk.stem.snowball import SnowballStemmer

# Загружаем модель
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
stemmer = SnowballStemmer("russian")

# Синонимы (взаимные, с поддержкой форм)
SYNONYMS = {
    "сим": ["симка", "симкарта", "сим-карта", "сим карта"],
    "кредитка": ["кредитная карта"],
    "наличные": ["наличка"]
}

# Ссылки на Excel-файлы
GITHUB_CSV_URLS = [
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data1.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data2.xlsx",
    "https://raw.githubusercontent.com/skatzrsk/semantic-assistant/main/data3.xlsx"
]

# Лёгкая предобработка текста
def preprocess(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

# Загрузка одного Excel-файла
def load_excel(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Ошибка загрузки {url}")
    df = pd.read_excel(BytesIO(response.content))

    topic_cols = [col for col in df.columns if col.lower().startswith("topics")]
    if not topic_cols:
        raise KeyError("Не найдены колонки topics")

    df = df[['phrase'] + topic_cols]
    df['topics'] = df[topic_cols].fillna('').agg(lambda x: [t for t in x.tolist() if t], axis=1)
    df['phrase_proc'] = df['phrase'].apply(preprocess)
    return df[['phrase', 'phrase_proc', 'topics']]

# Загрузка всех Excel-файлов
def load_all_excels():
    dfs = []
    for url in GITHUB_CSV_URLS:
        try:
            df = load_excel(url)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Ошибка с {url}: {e}")
    if not dfs:
        raise ValueError("Не удалось загрузить ни одного файла")
    return pd.concat(dfs, ignore_index=True)

# Семантический поиск
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

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]

# Точный поиск по ключевому слову (учёт синонимов и "/")
def keyword_search(query, df):
    query = preprocess(query)
    query_stem = stemmer.stem(query)

    # Расширяем список поисковых слов
    search_terms = set([query_stem])
    for key, synonyms in SYNONYMS.items():
        stems = {stemmer.stem(s) for s in synonyms + [key]}
        if query_stem in stems:
            search_terms.update(stems)

    matches = []
    for _, row in df.iterrows():
        parts = re.split(r"[\/|,]", row['phrase_proc'])
        for part in parts:
            words = re.findall(r'\b\w+\b', part)
            for word in words:
                if stemmer.stem(word) in search_terms:
                    matches.append((row['phrase'], row['topics']))
                    break
    return matches
