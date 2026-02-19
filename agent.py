

import json
import os
from typing import Optional, Callable
from dataclasses import asdict

from churn_analyzer import ChurnReport
from rag_engine import ChurnRAGEngine



class LLMProvider:
    def chat(self, messages: list[dict], system: str = "") -> str:
        raise NotImplementedError


class ClaudeProvider(LLMProvider):
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-haiku-4-5-20251001"):
        try:
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
            )
            self.model = model
            self._available = True
            print(f"✅ Claude provider: {model}")
        except Exception as e:
            print(f"⚠️  Claude недоступен: {e}")
            self._available = False

    def chat(self, messages: list[dict], system: str = "") -> str:
        if not self._available:
            raise RuntimeError("Claude API недоступен")
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system,
            messages=messages
        )
        return response.content[0].text


class OllamaProvider(LLMProvider):
    
    def __init__(self, model: str = "qwen2.5", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._available = self._check()

    def _check(self) -> bool:
        try:
            import requests
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            models = [m["name"] for m in r.json().get("models", [])]
            available = any(self.model in m for m in models)
            if available:
                print(f"✅ Ollama provider: {self.model}")
            else:
                print(f"⚠️  Ollama: модель {self.model} не найдена. "
                      f"Запустите: ollama pull {self.model}")
            return available
        except Exception:
            print("⚠️  Ollama недоступна. Запустите: ollama serve")
            return False

    def chat(self, messages: list[dict], system: str = "") -> str:
        import requests
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={"model": self.model, "messages": all_messages, "stream": False},
            timeout=120
        )
        response.raise_for_status()
        return response.json()["message"]["content"]



SYSTEM_PROMPT = """Ты — эксперт по анализу оттока клиентов (Customer Churn Analysis).
Ты помогаешь бизнесу понимать, почему уходят клиенты, и разрабатываешь стратегии удержания.

Ты можешь работать с любой индустрией: телеком, банки, SaaS, e-commerce, страхование.
Ты даёшь конкретные, actionable рекомендации на основе данных.
Ты объясняешь сложные ML-результаты простым языком.

При ответе структурируй информацию: 
- Ключевые факты из данных
- Интерпретация
- Рекомендации
- Следующие шаги

Будь кратким, но содержательным. Используй числа и факты."""


class ChurnAgent:
    
    def __init__(
        self,
        provider: str = "auto",
        ollama_model: str = "qwen2.5",
        claude_model: str = "claude-haiku-4-5-20251001",
        use_rag: bool = True
    ):
        self.llm = self._init_provider(provider, ollama_model, claude_model)
        self.rag = ChurnRAGEngine() if use_rag else None
        if self.rag:
            self.rag.initialize()
        self.conversation_history: list[dict] = []
        self.current_report: Optional[ChurnReport] = None

    def _init_provider(self, provider: str, ollama_model: str, claude_model: str) -> LLMProvider:
        if provider == "claude":
            return ClaudeProvider(model=claude_model)
        elif provider == "ollama":
            return OllamaProvider(model=ollama_model)
        elif provider == "auto":
            
            claude = ClaudeProvider(model=claude_model)
            if claude._available:
                return claude
            ollama = OllamaProvider(model=ollama_model)
            if ollama._available:
                return ollama
            print("⚠️  LLM недоступен. Используется режим без генерации.")
            return None
        raise ValueError(f"Неизвестный провайдер: {provider}")

    
    def analyze_report(self, report: ChurnReport) -> str:
       
        self.current_report = report

        
        rag_context = ""
        if self.rag:
            query = (
                f"отток {report.churn_rate}% "
                f"факторы: {', '.join(f['feature'] for f in report.top_factors[:3])}"
            )
            rag_context = self.rag.get_context(query, top_k=3)

        
        report_text = self._format_report_for_llm(report)

        user_message = f"""Проанализируй следующие результаты модели оттока клиентов:

{report_text}

{"--- Релевантный контекст из базы знаний ---" if rag_context else ""}
{rag_context}

Дай развёрнутый анализ:
1. Что говорят эти данные о поведении клиентов?
2. Какие факторы наиболее критичны?
3. Конкретные рекомендации для снижения оттока
4. Приоритет действий (что делать в первую очередь)"""

        response = self._call_llm(user_message)

        
        if self.rag:
            self.rag.add_analysis(
                f"Анализ датасета: churn_rate={report.churn_rate}%, "
                f"ROC-AUC={report.roc_auc}, "
                f"top_factor={report.top_factors[0]['feature'] if report.top_factors else 'N/A'}. "
                f"Анализ LLM: {response[:300]}...",
                metadata={"churn_rate": report.churn_rate, "roc_auc": report.roc_auc}
            )

        return response

    def chat(self, user_message: str) -> str:
        
        rag_context = ""
        if self.rag:
            rag_context = self.rag.get_context(user_message, top_k=2)

        
        context_parts = []
        if self.current_report:
            context_parts.append(
                f"Текущий датасет: {self.current_report.total_customers} клиентов, "
                f"отток {self.current_report.churn_rate}%, "
                f"ROC-AUC={self.current_report.roc_auc}"
            )
        if rag_context:
            context_parts.append(f"Контекст из базы знаний:\n{rag_context}")

        full_message = user_message
        if context_parts:
            full_message = "\n\n".join(context_parts) + f"\n\nВопрос: {user_message}"

        response = self._call_llm(full_message)
        return response

    def explain_customer(self, customer_data: dict, churn_prob: float) -> str:
       
        rag_context = ""
        if self.rag:
            query = " ".join(f"{k}={v}" for k, v in list(customer_data.items())[:5])
            rag_context = self.rag.get_context(query, top_k=2)

        message = f"""Клиент имеет следующие характеристики:
{json.dumps(customer_data, ensure_ascii=False, indent=2)}

Модель предсказала вероятность оттока: {churn_prob:.1f}%

{f"Релевантный контекст: {rag_context}" if rag_context else ""}

Объясни:
1. Почему у этого клиента такая вероятность оттока?
2. Какие факторы наиболее влияют?
3. Конкретные шаги для удержания именно этого клиента."""

        return self._call_llm(message)

    def generate_retention_strategy(self, segment_description: str) -> str:
        
        rag_context = ""
        if self.rag:
            rag_context = self.rag.get_context(
                f"стратегия удержания {segment_description}", top_k=3
            )

        message = f"""Разработай детальную стратегию удержания клиентов для следующего сегмента:

{segment_description}

{f"Лучшие практики из базы знаний:{chr(10)}{rag_context}" if rag_context else ""}

Стратегия должна включать:
1. Немедленные действия (первые 48 часов)
2. Краткосрочные меры (1-2 недели)
3. Долгосрочные изменения (1-3 месяца)
4. KPI для измерения эффективности
5. Примерный ROI от удержания"""

        return self._call_llm(message)

    
    def _call_llm(self, user_message: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_message})

        if self.llm is None:
            response = "[LLM недоступен] Запустите Ollama или укажите ANTHROPIC_API_KEY"
        else:
            try:
                response = self.llm.chat(
                    messages=self.conversation_history[-10:],  # последние 10 сообщений
                    system=SYSTEM_PROMPT
                )
            except Exception as e:
                response = f"Ошибка LLM: {e}"

        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def _format_report_for_llm(self, report: ChurnReport) -> str:
        lines = [
            f"📊 ОТЧЁТ ОБ ОТТОКЕ КЛИЕНТОВ",
            f"Всего клиентов: {report.total_customers:,}",
            f"Ушло: {report.churned_customers:,} ({report.churn_rate}%)",
            f"",
            f"КАЧЕСТВО МОДЕЛИ:",
            f"  Accuracy: {report.model_accuracy}%",
            f"  ROC-AUC: {report.roc_auc}",
            f"  Precision: {report.precision:.1%}",
            f"  Recall: {report.recall:.1%}",
            f"  F1: {report.f1:.1%}",
            f"",
            f"ТОП-5 ФАКТОРОВ ОТТОКА:",
        ]
        for i, factor in enumerate(report.top_factors[:5], 1):
            lines.append(f"  {i}. {factor['feature']}: {factor['importance']:.3f}")

        lines += [
            f"",
            f"КЛИЕНТЫ ВЫСОКОГО РИСКА:",
            f"  Количество: {report.high_risk_count:,}",
            f"  Реальный отток в группе: {report.high_risk_churn_rate}%",
            f"",
            f"СИСТЕМНЫЕ РЕКОМЕНДАЦИИ:",
        ]
        for rec in report.recommendations:
            lines.append(f"  • {rec}")

        return "\n".join(lines)

    def reset_conversation(self):
        self.conversation_history = []
        self.current_report = None



if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    provider = sys.argv[1] if len(sys.argv) > 1 else "auto"
    print(f"\n🤖 Запуск ChurnAgent (provider={provider})\n")

    agent = ChurnAgent(provider=provider)

    
    print("💬 Чат с агентом. Введите 'exit' для выхода.\n")
    while True:
        query = input("Вы: ").strip()
        if query.lower() in ("exit", "quit", "выход"):
            break
        if not query:
            continue
        response = agent.chat(query)
        print(f"\n🤖 Агент: {response}\n")