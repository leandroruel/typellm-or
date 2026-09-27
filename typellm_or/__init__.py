"""typellm_or — geração tipada estilo TypeLLM sobre a API do OpenRouter.

Uso:

    from typellm_or import TypeLLMClient

    client = TypeLLMClient(model="qwen/qwen3-vl-8b-instruct")
    result = client.generate(
        context="Leia o recibo anexado.",
        images=["recibo.jpg"],
        questions={
            "total": {"type": "number", "instructions": "Qual o valor total?"},
            "pago": {"type": "boolean", "instructions": "Está marcado como pago?"},
        },
    )
"""

from .client import TypeLLMClient, SchemaError

__all__ = ["TypeLLMClient", "SchemaError"]
