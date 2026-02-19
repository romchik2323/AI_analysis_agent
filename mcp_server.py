

import json
import sys
import os
import traceback
from pathlib import Path


try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️  mcp не установлен: pip install mcp", file=sys.stderr)

import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))

from churn_analyzer import UniversalChurnAnalyzer
from rag_engine import ChurnRAGEngine
from agent import ChurnAgent



_analyzer: UniversalChurnAnalyzer = None
_rag: ChurnRAGEngine = None
_agent: ChurnAgent = None


def get_analyzer() -> UniversalChurnAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = UniversalChurnAnalyzer()
        
        try:
            _analyzer.load_model()
        except FileNotFoundError:
            pass
    return _analyzer


def get_rag() -> ChurnRAGEngine:
    global _rag
    if _rag is None:
        _rag = ChurnRAGEngine()
        _rag.initialize()
    return _rag


def get_agent() -> ChurnAgent:
    global _agent
    if _agent is None:
        _agent = ChurnAgent(provider="auto", use_rag=True)
    return _agent



TOOLS = [
    {
        "name": "analyze_churn_dataset",
        "description": (
            "Загружает CSV-файл с данными клиентов, обучает модель машинного обучения "
            "и возвращает полный отчёт об оттоке: процент оттока, топ-факторы риска, "
            "качество модели (ROC-AUC, Accuracy), сегментный анализ и рекомендации. "
            "Работает с любой индустрией: телеком, банки, SaaS, e-commerce."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "csv_path": {
                    "type": "string",
                    "description": "Путь к CSV-файлу с данными клиентов"
                },
                "target_column": {
                    "type": "string",
                    "description": "Название целевой колонки (отток). По умолчанию 'auto' — автоопределение",
                    "default": "auto"
                },
                "model_type": {
                    "type": "string",
                    "enum": ["random_forest", "gradient_boosting"],
                    "description": "Тип модели ML. По умолчанию random_forest",
                    "default": "random_forest"
                }
            },
            "required": ["csv_path"]
        }
    },
    {
        "name": "score_customers",
        "description": (
            "Оценивает риск оттока для списка клиентов. "
            "Возвращает вероятность оттока (0-100%) и уровень риска (LOW/MEDIUM/HIGH) "
            "для каждого клиента. Требует предварительного обучения модели."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "customers": {
                    "type": "array",
                    "description": "Список клиентов в виде массива объектов с признаками",
                    "items": {"type": "object"}
                },
                "top_n": {
                    "type": "integer",
                    "description": "Вернуть только топ-N клиентов с наибольшим риском",
                    "default": 10
                }
            },
            "required": ["customers"]
        }
    },
    {
        "name": "get_retention_strategy",
        "description": (
            "Генерирует персонализированную стратегию удержания клиентов с помощью LLM. "
            "Использует RAG (базу знаний) для обогащения рекомендаций лучшими практиками. "
            "Поддерживает Claude API и локальный Qwen2.5."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "segment_description": {
                    "type": "string",
                    "description": "Описание сегмента клиентов для разработки стратегии удержания"
                }
            },
            "required": ["segment_description"]
        }
    },
    {
        "name": "search_knowledge_base",
        "description": (
            "Семантический поиск по базе знаний об оттоке клиентов. "
            "Использует E5-large эмбеддинги для поиска релевантных фактов, "
            "исторических анализов и лучших практик удержания."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Поисковый запрос (на русском или английском)"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Количество результатов",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "explain_customer_churn",
        "description": (
            "Объясняет вероятность оттока для конкретного клиента на понятном языке. "
            "Указывает ключевые факторы риска и персональные рекомендации по удержанию."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "customer_data": {
                    "type": "object",
                    "description": "Данные клиента в виде словаря признаков"
                }
            },
            "required": ["customer_data"]
        }
    },
    {
        "name": "get_model_status",
        "description": "Возвращает статус обученной модели: метрики, признаки, дата обучения.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    }
]



def handle_analyze_churn_dataset(args: dict) -> str:
    csv_path = args["csv_path"]
    target_column = args.get("target_column", "auto")
    model_type = args.get("model_type", "random_forest")

    if not Path(csv_path).exists():
        return json.dumps({"error": f"Файл не найден: {csv_path}"}, ensure_ascii=False)

    analyzer = get_analyzer()
    analyzer.load_data(csv_path, target_column=target_column)
    report = analyzer.train(model_name=model_type)

    
    try:
        agent = get_agent()
        llm_analysis = agent.analyze_report(report)
    except Exception as e:
        llm_analysis = f"LLM-анализ недоступен: {e}"

    result = {
        "status": "success",
        "report": {
            "total_customers": report.total_customers,
            "churned_customers": report.churned_customers,
            "churn_rate": f"{report.churn_rate}%",
            "model_accuracy": f"{report.model_accuracy}%",
            "roc_auc": report.roc_auc,
            "precision": f"{report.precision:.1%}",
            "recall": f"{report.recall:.1%}",
            "f1": f"{report.f1:.1%}",
            "top_factors": report.top_factors[:5],
            "high_risk_customers": report.high_risk_count,
            "high_risk_churn_rate": f"{report.high_risk_churn_rate}%",
            "recommendations": report.recommendations,
        },
        "llm_analysis": llm_analysis
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


def handle_score_customers(args: dict) -> str:
    customers = args["customers"]
    top_n = args.get("top_n", 10)

    analyzer = get_analyzer()
    if analyzer.model is None:
        return json.dumps({"error": "Модель не обучена. Сначала вызовите analyze_churn_dataset"})

    df = pd.DataFrame(customers)
    scored = analyzer.score_customers(df)

    result_records = scored.head(top_n)[
        [col for col in scored.columns if col not in ["churn_probability", "risk_level"]]
        + ["churn_probability", "risk_level"]
    ].to_dict(orient="records")

    
    for r in result_records:
        r["risk_level"] = str(r["risk_level"])

    return json.dumps({
        "status": "success",
        "total_scored": len(scored),
        "top_results": result_records
    }, ensure_ascii=False, indent=2)


def handle_get_retention_strategy(args: dict) -> str:
    segment_description = args["segment_description"]
    agent = get_agent()
    strategy = agent.generate_retention_strategy(segment_description)
    return json.dumps({
        "status": "success",
        "strategy": strategy
    }, ensure_ascii=False, indent=2)


def handle_search_knowledge_base(args: dict) -> str:
    query = args["query"]
    top_k = args.get("top_k", 3)
    rag = get_rag()
    results = rag.search(query, top_k=top_k)
    return json.dumps({
        "status": "success",
        "query": query,
        "results": results
    }, ensure_ascii=False, indent=2)


def handle_explain_customer_churn(args: dict) -> str:
    customer_data = args["customer_data"]
    analyzer = get_analyzer()

    if analyzer.model is None:
        return json.dumps({"error": "Модель не обучена"})

    scored = analyzer.score_single(customer_data)
    churn_prob = scored["churn_probability"]

    agent = get_agent()
    explanation = agent.explain_customer(customer_data, churn_prob)

    return json.dumps({
        "status": "success",
        "churn_probability": churn_prob,
        "risk_level": scored["risk_level"],
        "explanation": explanation
    }, ensure_ascii=False, indent=2)


def handle_get_model_status(args: dict) -> str:
    analyzer = get_analyzer()
    if analyzer.model is None:
        return json.dumps({"status": "no_model", "message": "Модель не обучена"})

    return json.dumps({
        "status": "ready",
        "feature_count": len(analyzer.feature_columns),
        "features": analyzer.feature_columns,
        "target_column": analyzer.target_column,
        "model_type": type(analyzer.model).__name__
    }, ensure_ascii=False, indent=2)


HANDLERS = {
    "analyze_churn_dataset": handle_analyze_churn_dataset,
    "score_customers": handle_score_customers,
    "get_retention_strategy": handle_get_retention_strategy,
    "search_knowledge_base": handle_search_knowledge_base,
    "explain_customer_churn": handle_explain_customer_churn,
    "get_model_status": handle_get_model_status,
}



if MCP_AVAILABLE:
    app = Server("churn-analyzer")

    @app.list_tools()
    async def list_tools():
        return [
            types.Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"]
            )
            for t in TOOLS
        ]

    @app.call_tool()
    async def call_tool(name: str, arguments: dict):
        if name not in HANDLERS:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": f"Неизвестный инструмент: {name}"})
            )]
        try:
            result = HANDLERS[name](arguments)
            return [types.TextContent(type="text", text=result)]
        except Exception as e:
            error_msg = json.dumps({
                "error": str(e),
                "traceback": traceback.format_exc()
            }, ensure_ascii=False)
            return [types.TextContent(type="text", text=error_msg)]

    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream, write_stream,
                app.create_initialization_options()
            )

    if __name__ == "__main__":
        import asyncio
        print("🚀 Churn Analysis MCP Server запущен", file=sys.stderr)
        asyncio.run(main())

else:
    
    if __name__ == "__main__":
        print("MCP SDK не установлен. Тест инструментов напрямую:")
        print(handle_get_model_status({}))