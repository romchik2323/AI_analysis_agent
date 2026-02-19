
import json
import numpy as np
import pickle
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings("ignore")




class SimpleVectorStore:
    

    def __init__(self):
        self.vectors: list[np.ndarray] = []
        self.documents: list[dict] = []

    def add(self, vector: np.ndarray, document: dict):
        self.vectors.append(vector / (np.linalg.norm(vector) + 1e-9))
        self.documents.append(document)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        if not self.vectors:
            return []
        q = query_vector / (np.linalg.norm(query_vector) + 1e-9)
        matrix = np.stack(self.vectors)
        scores = matrix @ q
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in top_indices:
            doc = dict(self.documents[i])
            doc["score"] = float(scores[i])
            results.append(doc)
        return results

    def __len__(self):
        return len(self.documents)




class E5Embedder:
    

    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._available = False
        self._try_load()

    def _try_load(self):
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
            self._torch = torch
            self._available = True
            print(f"✅ E5-large загружена: {self.model_name}")
        except Exception as e:
            print(f"⚠️  E5-large недоступна ({e}). Используется TF-IDF fallback.")
            self._available = False

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._available:
            return self._encode_e5(texts)
        return self._encode_tfidf(texts)

    def _encode_e5(self, texts: list[str]) -> np.ndarray:
        """E5 требует префикс 'query: ' или 'passage: '."""
        import torch
        prefixed = [f"passage: {t}" for t in texts]
        inputs = self._tokenizer(
            prefixed, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
        # Mean pooling
        token_embs = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        mask_expanded = attention_mask.unsqueeze(-1).float()
        embeddings = (token_embs * mask_expanded).sum(1) / mask_expanded.sum(1)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.numpy()

    def encode_query(self, text: str) -> np.ndarray:
        """Кодирует запрос с префиксом 'query: '."""
        if self._available:
            import torch
            inputs = self._tokenizer(
                f"query: {text}", return_tensors="pt",
                truncation=True, max_length=512
            )
            with torch.no_grad():
                outputs = self._model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            return emb.squeeze().numpy()
        return self._encode_tfidf([text])[0]

    def _encode_tfidf(self, texts: list[str]) -> np.ndarray:
        """Fallback: простой TF-IDF вектор."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        if not hasattr(self, "_tfidf"):
            self._tfidf = TfidfVectorizer(max_features=512)
            self._tfidf.fit(texts)
        try:
            return self._tfidf.transform(texts).toarray().astype(np.float32)
        except Exception:
            self._tfidf.fit(texts)
            return self._tfidf.transform(texts).toarray().astype(np.float32)



class ChurnRAGEngine:
   
    DOMAIN_KNOWLEDGE = [
        {
            "type": "domain",
            "text": "Клиенты с помесячными контрактами уходят в 3-4 раза чаще, "
                    "чем клиенты с годовыми или двухлетними контрактами. "
                    "Стратегия удержания: переводить на долгосрочные контракты со скидкой.",
            "tags": ["contract", "retention"]
        },
        {
            "type": "domain",
            "text": "Оптоволоконный интернет (Fiber optic) ассоциируется с высоким оттоком ~42%. "
                    "Возможные причины: высокая цена, технические проблемы, конкуренция. "
                    "Рекомендация: улучшить качество сервиса и пересмотреть тарифы.",
            "tags": ["internet", "fiber", "pricing"]
        },
        {
            "type": "domain",
            "text": "Отсутствие техподдержки увеличивает отток с 15% до 42%. "
                    "Техподдержка — один из самых сильных удерживающих факторов. "
                    "Бесплатная техподдержка для новых клиентов снижает отток на 61%.",
            "tags": ["techsupport", "retention"]
        },
        {
            "type": "domain",
            "text": "Электронные чеки (electronic check) связаны с 45% оттока. "
                    "Это может указывать на низкую приверженность и финансовые трудности. "
                    "Клиенты на автоплатёже (кредитная карта, банковский перевод) уходят реже.",
            "tags": ["payment", "electronic_check"]
        },
        {
            "type": "domain",
            "text": "Tenure (время с компанией) — самый важный предиктор лояльности. "
                    "Клиенты первых 6 месяцев — группа максимального риска (onboarding churn). "
                    "Программы onboarding и early engagement критически важны.",
            "tags": ["tenure", "onboarding", "loyalty"]
        },
        {
            "type": "domain",
            "text": "SaaS-компании: отток >5% в месяц критичен для роста. "
                    "Ключевые причины: плохой onboarding, отсутствие value realization, "
                    "конкуренты, изменение потребностей. NPS < 30 коррелирует с высоким оттоком.",
            "tags": ["saas", "metrics"]
        },
        {
            "type": "domain",
            "text": "Банковские клиенты: отток коррелирует с количеством продуктов. "
                    "Клиенты с 1 продуктом уходят в 2-3 раза чаще, чем с 3+. "
                    "Cross-sell и up-sell — стратегии удержания в banking.",
            "tags": ["banking", "cross-sell"]
        },
        {
            "type": "domain",
            "text": "E-commerce: отток можно предсказать по RFM-метрикам. "
                    "Recency > 90 дней — сигнал тревоги. "
                    "Персонализированные предложения возвращают 20-30% dormant-клиентов.",
            "tags": ["ecommerce", "rfm"]
        },
        {
            "type": "strategy",
            "text": "Стратегия удержания: сегментируй клиентов по вероятности оттока, "
                    "рассчитай LTV каждого сегмента, оптимизируй бюджет на удержание "
                    "пропорционально LTV × вероятность_оттока.",
            "tags": ["strategy", "ltv", "segmentation"]
        },
        {
            "type": "strategy",
            "text": "Проактивное удержание эффективнее реактивного. "
                    "Контактировать с клиентом нужно до того, как он принял решение уйти. "
                    "Оптимальное окно: 2-4 недели после сигнала риска.",
            "tags": ["strategy", "proactive", "timing"]
        },
    ]

    def __init__(self, store_path: str = "rag_store.pkl"):
        self.store_path = Path(store_path)
        self.embedder = E5Embedder()
        self.store = SimpleVectorStore()
        self._initialized = False

    def initialize(self):
        
        if self.store_path.exists():
            self._load_store()
        else:
            self._build_initial_store()
        self._initialized = True
        print(f"🔍 RAG Engine готов: {len(self.store)} документов в базе")

    def _build_initial_store(self):
        texts = [doc["text"] for doc in self.DOMAIN_KNOWLEDGE]
        embeddings = self.embedder.encode(texts)
        for doc, emb in zip(self.DOMAIN_KNOWLEDGE, embeddings):
            self.store.add(emb, doc)
        self._save_store()

    def _save_store(self):
        with open(self.store_path, "wb") as f:
            pickle.dump(self.store, f)

    def _load_store(self):
        with open(self.store_path, "rb") as f:
            self.store = pickle.load(f)

    def add_analysis(self, analysis_text: str, metadata: dict = None):
        
        doc = {
            "type": "analysis",
            "text": analysis_text,
            "metadata": metadata or {}
        }
        emb = self.embedder.encode([analysis_text])[0]
        self.store.add(emb, doc)
        self._save_store()

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        
        if not self._initialized:
            self.initialize()
        query_emb = self.embedder.encode_query(query)
        results = self.store.search(query_emb, top_k=top_k)
        return results

    def get_context(self, query: str, top_k: int = 3) -> str:
        
        docs = self.search(query, top_k=top_k)
        if not docs:
            return ""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            score = doc.get("score", 0)
            context_parts.append(
                f"[Знание {i} | релевантность: {score:.2f}]\n{doc['text']}"
            )
        return "\n\n".join(context_parts)