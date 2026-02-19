"""
FastAPI Backend — REST API для системы скоринга оттока.
Эндпоинты:
  POST /train          — обучить модель на CSV
  POST /score          — скоринг списка клиентов
  POST /score/single   — скоринг одного клиента
  POST /chat           — чат с LLM-агентом
  GET  /health         — статус сервиса
  GET  /model/status   — информация о модели
  POST /knowledge/add  — добавить знание в RAG
  POST /knowledge/search — поиск по базе знаний

Запуск:
    uvicorn api:app --reload --port 8000
"""

import os
import sys
from pathlib import Path
from typing import Optional, Any
import json

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import uvicorn

sys.path.insert(0, str(Path(__file__).parent))

from churn_analyzer import UniversalChurnAnalyzer
from rag_engine import ChurnRAGEngine
from agent import ChurnAgent



app = FastAPI(
    title="Churn Analysis API",
    description=(
        "Universal Customer Churn Prediction API. "
        "Поддерживает телеком, банки, SaaS, e-commerce. "
        "LLM-агент: Claude API + Qwen2.5 (Ollama). RAG на E5-large."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


_analyzer: Optional[UniversalChurnAnalyzer] = None
_agent: Optional[ChurnAgent] = None
_rag: Optional[ChurnRAGEngine] = None
_last_report = None


def get_analyzer() -> UniversalChurnAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = UniversalChurnAnalyzer()
        try:
            _analyzer.load_model()
        except FileNotFoundError:
            pass
    return _analyzer


def get_agent() -> ChurnAgent:
    global _agent
    if _agent is None:
        provider = os.getenv("LLM_PROVIDER", "auto")
        ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5")
        _agent = ChurnAgent(provider=provider, ollama_model=ollama_model)
    return _agent


def get_rag() -> ChurnRAGEngine:
    global _rag
    if _rag is None:
        _rag = ChurnRAGEngine()
        _rag.initialize()
    return _rag



class TrainRequest(BaseModel):
    csv_path: str = Field(..., description="Путь к CSV-файлу")
    target_column: str = Field("auto", description="Целевая колонка")
    model_type: str = Field("random_forest", description="random_forest | gradient_boosting")
    use_llm_analysis: bool = Field(True, description="Запустить LLM-анализ результатов")


class ScoreRequest(BaseModel):
    customers: list[dict[str, Any]] = Field(..., description="Список клиентов")
    top_n: int = Field(10, description="Вернуть топ-N по риску")
    explain_top: int = Field(0, description="Объяснить топ-N клиентов через LLM")


class SingleScoreRequest(BaseModel):
    customer: dict[str, Any] = Field(..., description="Данные клиента")
    explain: bool = Field(False, description="Сгенерировать LLM-объяснение")


class ChatRequest(BaseModel):
    message: str = Field(..., description="Вопрос к агенту")
    reset: bool = Field(False, description="Сбросить историю диалога")


class KnowledgeAddRequest(BaseModel):
    text: str = Field(..., description="Текст для добавления в базу знаний")
    metadata: dict = Field(default_factory=dict)


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(..., description="Поисковый запрос")
    top_k: int = Field(3)


class RetentionStrategyRequest(BaseModel):
    segment_description: str = Field(..., description="Описание сегмента клиентов")



@app.get("/health")
async def health():
    analyzer = get_analyzer()
    return {
        "status": "ok",
        "model_loaded": analyzer.model is not None,
        "rag_documents": len(get_rag().store),
        "version": "2.0.0"
    }


@app.get("/model/status")
async def model_status():
    analyzer = get_analyzer()
    if analyzer.model is None:
        return {"status": "no_model", "message": "Модель не обучена"}
    return {
        "status": "ready",
        "features": analyzer.feature_columns,
        "feature_count": len(analyzer.feature_columns),
        "target_column": analyzer.target_column,
        "model_type": type(analyzer.model).__name__
    }


@app.post("/train")
async def train(request: TrainRequest, background_tasks: BackgroundTasks):
    global _last_report

    if not Path(request.csv_path).exists():
        raise HTTPException(404, f"Файл не найден: {request.csv_path}")

    analyzer = get_analyzer()
    try:
        analyzer.load_data(request.csv_path, target_column=request.target_column)
        report = analyzer.train(model_name=request.model_type)
        _last_report = report
    except Exception as e:
        raise HTTPException(400, str(e))

    response_data = {
        "status": "success",
        "report": {
            "total_customers": report.total_customers,
            "churned_customers": report.churned_customers,
            "churn_rate": f"{report.churn_rate}%",
            "model_accuracy": f"{report.model_accuracy}%",
            "roc_auc": report.roc_auc,
            "top_factors": report.top_factors[:5],
            "high_risk_count": report.high_risk_count,
            "recommendations": report.recommendations,
        },
        "llm_analysis": None
    }

    if request.use_llm_analysis:
        try:
            agent = get_agent()
            response_data["llm_analysis"] = agent.analyze_report(report)
        except Exception as e:
            response_data["llm_analysis"] = f"LLM недоступен: {e}"

    return JSONResponse(content=response_data)


@app.post("/score")
async def score(request: ScoreRequest):
    analyzer = get_analyzer()
    if analyzer.model is None:
        raise HTTPException(400, "Модель не обучена. Вызовите /train сначала.")

    df = pd.DataFrame(request.customers)
    scored = analyzer.score_customers(df)

    results = []
    for _, row in scored.head(request.top_n).iterrows():
        record = row.to_dict()
        record["risk_level"] = str(record["risk_level"])
        results.append(record)

    
    if request.explain_top > 0:
        agent = get_agent()
        for i, record in enumerate(results[:request.explain_top]):
            try:
                customer_data = {k: v for k, v in record.items()
                                 if k not in ("churn_probability", "risk_level")}
                record["llm_explanation"] = agent.explain_customer(
                    customer_data, record["churn_probability"]
                )
            except Exception as e:
                record["llm_explanation"] = f"Ошибка: {e}"

    return {
        "status": "success",
        "total_scored": len(scored),
        "results": results
    }


@app.post("/score/single")
async def score_single(request: SingleScoreRequest):
    analyzer = get_analyzer()
    if analyzer.model is None:
        raise HTTPException(400, "Модель не обучена")

    scored = analyzer.score_single(request.customer)
    response = {"status": "success", **scored}

    if request.explain:
        try:
            agent = get_agent()
            response["explanation"] = agent.explain_customer(
                request.customer, scored["churn_probability"]
            )
        except Exception as e:
            response["explanation"] = f"Ошибка LLM: {e}"

    return response


@app.post("/chat")
async def chat(request: ChatRequest):
    agent = get_agent()

    if request.reset:
        agent.reset_conversation()
        if _last_report:
            agent.current_report = _last_report

    try:
        response = agent.chat(request.message)
        return {
            "status": "success",
            "response": response,
            "history_length": len(agent.conversation_history)
        }
    except Exception as e:
        raise HTTPException(500, f"Ошибка агента: {e}")


@app.post("/retention-strategy")
async def retention_strategy(request: RetentionStrategyRequest):
    agent = get_agent()
    strategy = agent.generate_retention_strategy(request.segment_description)
    return {"status": "success", "strategy": strategy}


@app.post("/knowledge/add")
async def knowledge_add(request: KnowledgeAddRequest):
    rag = get_rag()
    rag.add_analysis(request.text, metadata=request.metadata)
    return {"status": "success", "total_documents": len(rag.store)}


@app.post("/knowledge/search")
async def knowledge_search(request: KnowledgeSearchRequest):
    rag = get_rag()
    results = rag.search(request.query, top_k=request.top_k)
    return {"status": "success", "query": request.query, "results": results}



if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Churn Analysis API запущен на http://localhost:{port}")
    print(f"📚 Документация: http://localhost:{port}/docs")
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)
