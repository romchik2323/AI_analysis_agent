#  AI Analysis Agent

**Python • LLM • MCP • RAG • FastAPI • Machine Learning**

Агентная система предсказания и анализа оттока клиентов с поддержкой Claude API, локального Qwen2.5 и семантическим поиском на E5-large. Универсальный ML работает с любым табличным датасетом.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)


---

##  Задача проекта

Компании теряют 20-30% клиентов ежегодно. Чтобы удерживать клиентов, нужно:

1. **Предсказать отток** до того, как клиент уйдёт
2. **Понять причины** — какие факторы влияют на решение уйти
3. **Разработать стратегию** — персонализированные действия для удержания
4. **Объяснить модель** — не просто цифры, а понятный анализ для бизнеса

Эта система решает все 4 задачи, работая как **AI-ассистент** для аналитиков и менеджеров.

---
##  Что применено

### Machine Learning
- Feature engineering для табличных данных
- Обработка несбалансированных классов (`class_weight='balanced'`)
- Feature importance анализ
- Cross-validation и метрики (ROC-AUC, Precision/Recall)

### LLM & NLP
- Работа с Anthropic API (функции, streaming)
- Интеграция локальных моделей через Ollama
- Prompt engineering для аналитических задач
- Context window management (обрезка истории)

### RAG (Retrieval-Augmented Generation)
- Sentence embeddings (E5-large, 1024-dim)
- Векторный поиск (cosine similarity)
- Префиксы для E5 (`query:` vs `passage:`)
- Гибридный поиск (semantic + metadata filters)

### Backend Development
- FastAPI асинхронные эндпоинты
- Pydantic схемы валидации
- REST API design (CRUD + поиск)
- Background tasks для долгих операций

### MCP (Model Context Protocol)
- Реализация MCP сервера
- Определение инструментов (tools)
- stdio транспорт
- Интеграция с Claude Desktop / Cursor

### Software Engineering
- Модульная архитектура (каждый компонент независим)
- Dependency injection (провайдеры LLM)
- Error handling и fallback механизмы
- Сериализация моделей (pickle)
- Type hints и dataclasses

##  Архитектура

```
┌─────────────────────────────────────────────────────────┐
│              MCP Server / FastAPI Backend               │
│        (Claude Desktop, Cursor, REST API clients)       │
└──────────────┬──────────────────────────┬───────────────┘
               │                          │
    ┌──────────▼──────────┐   ┌───────────▼───────────┐
    │    LLM Agent        │   │   Universal Churn     │
    │                     │   │      Analyzer         │
    │  • Claude API       │   │                       │
    │  • Qwen2.5 (Ollama) │   │  • Random Forest      │
    │  • Auto provider    │   │  • Gradient Boosting  │
    │                     │   │  • Auto preprocessing │
    └──────────┬──────────┘   └───────────┬───────────┘
               │                          │
    ┌──────────▼──────────────────────────▼───────────┐
    │              RAG Engine                         │
    │  • E5-large embeddings (1024-dim)               │
    │  • Vector store (numpy/pickle)                  │
    │  • 10+ domain knowledge rules                   │
    │  • Historical analysis storage                  │
    └─────────────────────────────────────────────────┘
```

**Ключевая особенность:** полная модульность — каждый компонент работает независимо и может использоваться отдельно.

---

##  Возможности

### 1️⃣ Universal ML Pipeline

```python
from churn_analyzer import UniversalChurnAnalyzer

# Работает с ЛЮБЫМ CSV — автоопределение структуры
analyzer = UniversalChurnAnalyzer()
analyzer.load_data("any_dataset.csv")  # автоматически находит целевую колонку
report = analyzer.train()  # обучает модель за 1 вызов

print(f"ROC-AUC: {report.roc_auc}")
print(f"Top risk factor: {report.top_factors[0]['feature']}")
```

**Индустрии:**
-  Телеком (отток абонентов)
-  Банки (закрытие счетов)
-  SaaS (отмена подписок)
-  E-commerce (потеря покупателей)
-  Страхование (отказ от полиса)

**Автоматически обрабатывает:**
- Категориальные признаки → LabelEncoder
- Пропуски → заполнение медианой/Unknown
- ID-колонки → автоудаление
- Целевая переменная → бинаризация (Yes/No → 1/0)

---

### 2️⃣ LLM Agent с RAG

```python
from agent import ChurnAgent

# Автовыбор: Claude API → Ollama → CPU-only mode
agent = ChurnAgent(provider="auto")

# Анализ отчёта с контекстом из базы знаний
analysis = agent.analyze_report(report)
print(analysis)
# → "Основная причина оттока — помесячные контракты (42.7% vs 2.8% для 
#    двухлетних). Рекомендуется программа лояльности с переводом на 
#    долгосрочные тарифы со скидкой 15%..."

# Диалог
response = agent.chat("Почему клиенты с Fiber optic уходят чаще?")
# → Система ищет в RAG базе знаний, находит релевантные факты, 
#    генерирует ответ с учётом контекста

# Персональное объяснение
customer = {"tenure": 2, "Contract": "Month-to-month", "MonthlyCharges": 85}
explanation = agent.explain_customer(customer, churn_prob=67.3)
# → "Этот клиент в группе высокого риска из-за короткого срока (2 месяца) 
#    и помесячного контракта. Рекомендация: предложить upgrade на..."
```

**LLM провайдеры:**
- ✅ **Claude API** (Anthropic) — лучшее качество анализа
- ✅ **Qwen2.5** (Ollama) — полный offline режим, бесплатно
- ✅ **Auto mode** — автовыбор доступного провайдера

---

### 3️⃣ RAG Engine на E5-large

```python
from rag_engine import ChurnRAGEngine

rag = ChurnRAGEngine()
rag.initialize()

# Семантический поиск по базе знаний
results = rag.search("почему помесячные контракты опасны для бизнеса")
# → [
#     {"text": "Клиенты с помесячными контрактами уходят в 3-4 раза чаще...",
#      "score": 0.89},
#     {"text": "Стратегия удержания: переводить на долгосрочные...",
#      "score": 0.76}
# ]

# Добавление своих знаний
rag.add_analysis(
    "Анализ Q4 2025: основной фактор оттока в декабре — плохое качество поддержки",
    metadata={"quarter": "Q4_2025", "industry": "telecom"}
)
```

**База знаний включает:**
- Доменные правила об оттоке (телеком, SaaS, банки, e-commerce)
- Стратегии удержания (проактивные vs реактивные)
- Best practices из индустрии
- Исторические анализы (добавляются автоматически)

**Технология:** `intfloat/multilingual-e5-large` — state-of-the-art многоязычная модель эмбеддингов (1024-dim vectors).

---

### 4️⃣ MCP Server для Claude Desktop

```bash
# Установка
python mcp_server.py
```

Добавьте в `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "churn-analyzer": {
      "command": "python",
      "args": ["/path/to/mcp_server.py"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

После этого в **Claude Desktop** появятся инструменты:

| Инструмент | Описание |
|-----------|----------|
| `analyze_churn_dataset` | Обучить модель на любом CSV, получить полный отчёт |
| `score_customers` | Скоринг списка клиентов с вероятностями оттока |
| `get_retention_strategy` | Сгенерировать стратегию удержания для сегмента |
| `search_knowledge_base` | Поиск по базе знаний об оттоке |
| `explain_customer_churn` | Объяснение риска для конкретного клиента |
| `get_model_status` | Проверка состояния модели |

**Пример диалога с Claude Desktop:**
```
User: Проанализируй отток клиентов из файла /Users/data/customers.csv

Claude: [использует analyze_churn_dataset]
        Обучил модель на 7,043 клиентах. ROC-AUC: 0.91, точность 85%.
        
        Ключевые находки:
        • Помесячные контракты → 42.7% оттока (в 3.2 раза выше нормы)
        • Fiber optic → 41.9% оттока (проблемы с ценой/качеством)
        • Отсутствие техподдержки → +61% к риску
        
        Рекомендации:
        1. Программа лояльности для перевода на долгосрочные контракты
        2. Пересмотр ценообразования на Fiber optic
        3. Бесплатная техподдержка для новых клиентов
```

---

### 5️⃣ FastAPI REST Backend

```bash
# Запуск
uvicorn api:app --reload --port 8000
# Документация: http://localhost:8000/docs
```

**9 эндпоинтов:**

```bash
# Обучить модель
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "csv_path": "customers.csv",
    "use_llm_analysis": true
  }'

# Скоринг одного клиента
curl -X POST http://localhost:8000/score/single \
  -d '{
    "customer": {
      "tenure": 3,
      "Contract": "Month-to-month",
      "MonthlyCharges": 75,
      "InternetService": "Fiber optic"
    },
    "explain": true
  }'
# Response:
# {
#   "churn_probability": 64.2,
#   "risk_level": "🔴 HIGH",
#   "explanation": "Клиент в зоне риска из-за короткого tenure (3 мес) и..."
# }

# Чат с агентом
curl -X POST http://localhost:8000/chat \
  -d '{"message": "Какие клиенты уходят чаще всего?"}'

# Стратегия удержания
curl -X POST http://localhost:8000/retention-strategy \
  -d '{"segment_description": "Клиенты с Fiber optic и помесячным контрактом"}'
```

---

##  Результаты (Telco Dataset)

### Базовая статистика

| Показатель | Значение |
|-----------|----------|
| Всего клиентов | 7,043 |
| Ушло клиентов | 1,869 |
| **Процент оттока** | **26.5%** |
| Средний tenure | 32.4 месяцев |
| Средний платёж | $64.76/мес |

### Качество модели

| Метрика | Random Forest | Gradient Boosting |
|---------|--------------|-------------------|
| **Accuracy** | **85%** | 83% |
| **ROC-AUC** | **0.91** | 0.89 |
| Precision | 65% | 68% |
| Recall | 78% | 73% |
| F1 Score | 70% | 70% |

### Топ-5 факторов оттока

```
1. Tenure (время с компанией)       → 0.242 importance
2. MonthlyCharges (платёж/мес)      → 0.186 importance
3. Contract (тип контракта)         → 0.134 importance
4. InternetService (тип интернета)  → 0.092 importance
5. PaymentMethod (способ оплаты)    → 0.071 importance
```

### Сегментный анализ

| Сегмент | Отток | Интерпретация |
|---------|-------|---------------|
| **Month-to-month контракт** | **42.7%** | В 3.2 раза выше среднего |
| One year контракт | 11.3% | Норма |
| Two year контракт | 2.8% | Лояльные клиенты |
| **Fiber optic интернет** | **41.9%** | Проблема ценообразования |
| DSL интернет | 19.0% | Норма |
| Без техподдержки | 41.6% | Критично |
| С техподдержкой | 15.2% | Снижение риска на 61% |

### Группа высокого риска

**Профиль:** Помесячный контракт + Fiber optic + Без техподдержки

- **Размер группы:** 1,796 клиентов (25.5%)
- **Реальный отток:** **57.5%**
- **Средний платёж:** $85.49/мес
- **LTV под угрозой:** $1,836,000/год

**Рекомендации:**
1. Персональные предложения на переход на годовой контракт (-20% discount)
2. Бесплатная техподдержка на 3 месяца
3. Аудит качества Fiber optic в регионе

---

##  Быстрый старт

### Установка

```bash
git clone https://github.com/romchik2323/AI_analysis_agent
cd churn-analysis-agent
pip install -r requirements.txt
```

### 1. Простой ML без LLM (работает сразу)

```python
from churn_analyzer import UniversalChurnAnalyzer

analyzer = UniversalChurnAnalyzer()
analyzer.load_data("WA_Fn-UseC_-Telco-Customer-Churn.csv")
report = analyzer.train()

print(f"ROC-AUC: {report.roc_auc}")
print(f"Отток: {report.churn_rate}%")
print(f"Группа риска: {report.high_risk_count} клиентов")
```

### 2. С локальным Qwen2.5 (offline)

```bash
# Установить Ollama
brew install ollama  # macOS
# или https://ollama.ai для Windows/Linux

# Скачать модель
ollama pull qwen2.5

# Запуск
python -c "
from churn_analyzer import UniversalChurnAnalyzer
from agent import ChurnAgent

analyzer = UniversalChurnAnalyzer()
analyzer.load_data('data.csv')
report = analyzer.train()

agent = ChurnAgent(provider='ollama')
print(agent.analyze_report(report))
"
```

### 3. С Claude API (лучшее качество)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

python -c "
from agent import ChurnAgent
agent = ChurnAgent(provider='claude')
response = agent.chat('Какие стратегии удержания работают лучше всего?')
print(response)
"
```

### 4. REST API сервер

```bash
uvicorn api:app --reload --port 8000
# Откройте: http://localhost:8000/docs
```

### 5. MCP для Claude Desktop

См. секцию "MCP Server" выше — после настройки просто общайтесь с Claude в Desktop приложении.

---

## 🛠️ Технологии

### Core Stack

```python
pandas>=2.0.0              # Обработка данных
numpy>=1.24.0              # Вычисления
scikit-learn>=1.3.0        # ML модели
```

### LLM & Embeddings

```python
anthropic>=0.25.0          # Claude API
requests>=2.31.0           # Ollama HTTP client
transformers>=4.40.0       # E5-large для RAG
torch>=2.0.0               # PyTorch backend
sentence-transformers      # Удобный API для embeddings
```

### API & MCP

```python
fastapi>=0.110.0           # REST API
uvicorn[standard]>=0.29.0  # ASGI сервер
pydantic>=2.0.0            # Валидация данных
mcp>=1.0.0                 # Model Context Protocol SDK
```

### Database (опционально)

```python
sqlalchemy>=2.0.0          # ORM для баз данных
psycopg2-binary>=2.9.0     # PostgreSQL драйвер
```

---

##  Структура проекта

```
churn-analysis-agent/
│
├── churn_analyzer.py       # Universal ML pipeline (любой CSV)
├── agent.py                # LLM Agent (Claude + Ollama/Qwen2.5)
├── rag_engine.py           # RAG на E5-large embeddings
├── mcp_server.py           # MCP Server для Claude Desktop
├── api.py                  # FastAPI REST Backend
│
├── requirements.txt        # Зависимости
├── README.md              # Этот файл
└── КАК_ЗАПУСТИТЬ.md       # Детальная инструкция для новичков
│
└── models/                # Сохранённые модели
    ├── churn_model.pkl    # Обученная ML-модель
    └── rag_store.pkl      # Векторное хранилище RAG
```

---

##  Как работает

### ML Pipeline

```python
def train(csv_path: str):
    # 1. Загрузка
    df = pd.read_csv(csv_path)
    
    # 2. Автопрепроцессинг
    target = auto_detect_target_column(df)  # churn, churned, label, etc.
    X, y = preprocess(df, target)
    # → Категории → LabelEncoder
    # → Пропуски → заполнение
    # → ID-колонки → удаление
    
    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )
    
    # 4. Обучение
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',  # для несбалансированных данных
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 5. Оценка
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'feature_importance': model.feature_importances_
    }
    
    return model, metrics
```

### RAG Pipeline

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Инициализация
model = SentenceTransformer('intfloat/multilingual-e5-large')

# 2. Индексация базы знаний
knowledge = [
    "Клиенты с помесячными контрактами уходят в 3-4 раза чаще...",
    "Техподдержка снижает отток на 61%...",
    # ... 10+ правил
]
embeddings = model.encode([f"passage: {k}" for k in knowledge])

# 3. Поиск
query = "почему клиенты уходят с Fiber optic"
query_emb = model.encode(f"query: {query}")

# Cosine similarity
similarities = np.dot(embeddings, query_emb) / (
    np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb)
)
top_k = np.argsort(similarities)[::-1][:3]

results = [knowledge[i] for i in top_k]
# → Передаём в LLM как контекст
```

### LLM Agent Flow

```python
def analyze_report(report):
    # 1. RAG: найти релевантный контекст
    context = rag.search(
        f"отток {report.churn_rate}% "
        f"факторы: {', '.join(report.top_factors[:3])}"
    )
    
    # 2. Формирование промпта
    prompt = f"""
    Проанализируй результаты модели оттока:
    
    Данные: {format_report(report)}
    
    Контекст из базы знаний: {context}
    
    Дай анализ: причины, рекомендации, приоритет действий.
    """
    
    # 3. LLM генерация
    if provider == "claude":
        response = anthropic_client.messages.create(
            model="claude-sonnet-4",
            messages=[{"role": "user", "content": prompt}]
        )
    elif provider == "ollama":
        response = ollama_client.chat(model="qwen2.5", messages=...)
    
    # 4. Сохранение в RAG для будущих запросов
    rag.add_analysis(response, metadata={"dataset": "telco"})
    
    return response
```

##  Расширение функционала

### Добавить новый LLM провайдер

```python
# agent.py

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4"):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def chat(self, messages: list[dict], system: str = "") -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}] + messages
        )
        return response.choices[0].message.content
```

### Добавить новую модель ML

```python
# churn_analyzer.py

from xgboost import XGBClassifier

def train(self, model_name: str = "random_forest"):
    if model_name == "xgboost":
        self.model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6
        )
    # ... остальной код
```

### Добавить custom метрики

```python
def calculate_custom_metrics(y_true, y_pred, y_proba):
    from sklearn.metrics import fbeta_score
    
    # F2 score (recall важнее precision)
    f2 = fbeta_score(y_true, y_pred, beta=2)
    
    # Profit curve
    threshold_profits = []
    for threshold in np.linspace(0, 1, 100):
        pred = (y_proba >= threshold).astype(int)
        # Предположим: удержание стоит $50, упущенный клиент стоит $500
        profit = (pred * y_true * 450 - pred * (1-y_true) * 50).sum()
        threshold_profits.append(profit)
    
    optimal_threshold = np.argmax(threshold_profits) / 100
    
    return {"f2": f2, "optimal_threshold": optimal_threshold}
```

---

##  Ограничения

1. **Размер данных:** Оптимален для датасетов 1,000-100,000 строк. Для больших нужна batch-обработка.

2. **Только табличные данные:** Не работает с текстом, изображениями, временными рядами напрямую.

3. **RAG база знаний:** Пока 10 правил — для production нужно 100+.

4. **LLM токены:** При использовании Claude API учитывайте стоимость токенов для больших отчётов.

5. **Объяснимость:** Feature importance показывает какие признаки важны, но не как именно они влияют (для этого нужен SHAP/LIME).

---

##  Roadmap

- [ ] SHAP values для объяснения предсказаний
- [ ] Поддержка временных рядов (LSTM для sequence churn)
- [ ] Web UI (Streamlit/Gradio)
- [ ] A/B тест симулятор для retention стратегий
- [ ] Интеграция с CRM системами (Salesforce, HubSpot)
- [ ] Real-time scoring API (streaming predictions)
- [ ] Multi-tenant поддержка (разные компании на одном сервере)





