import pandas as pd
import requests
import re
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
from pymorphy2 import MorphAnalyzer

# Инициализация модели и морфоанализатора
morph = MorphAnalyzer()
model = SentenceTransformer("intfloat/multilingual-e5-large")  # Новая мощная модель

# Ссылки на Excel-файлы
GITHUB_CSV_URLS = [
    "https://raw.githubusercontent.com/d3ld3l/semantic-assistant/main/data1.xlsx",
    "https://raw.githubusercontent.com/d3ld3l/semantic-assistant/main/data2.xlsx",
    "https://raw.githubusercontent.com/d3ld3l/semantic-assistant/main/data3.xlsx"
]

# Словарь синонимов
SYNONYMS = {
    "симкарта": ["сим", "симка", "сим-карта", "сим карта", "симки", "симкарты", "сим-карты", "симкарте", "сим-карту"],
    "карта": ["карточка", "картой", "карточку"],
    "утеряна": ["потеряна", "потерял", "пропала"],
    "оплатить": ["совершить оплату", "провести оплату", "произвести оплату", "осуществить оплату"],
    "не могу": ["не получается", "не проводится", "не производится", "не осуществляется", "не проходит"],
    "не проходит оплата": ["не получается оплатить", "не проводится оплата", "не производится оплата", "не осуществляется оплата", "не могу совершить оплату", "не совершается оплата"],
    "какую сумму": ["сколько"]
}

# Построение обратного отображения для синони
