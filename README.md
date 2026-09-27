# typellm-or

Geração tipada estilo [TypeLLM](https://github.com/TypeLLM/TypeLLM), mas rodando em qualquer modelo (inclusive vision) do [OpenRouter](https://openrouter.ai) — sem precisar de GPU local nem servidor SGLang.

Os tipos declarados são garantidos via structured outputs (`json_schema` strict), o equivalente ao constrained decoding do TypeLLM original.

## Instalação

```bash
pip install openai
export OPENROUTER_API_KEY=sk-or-...
```

## Uso

```python
from typellm_or import TypeLLMClient

client = TypeLLMClient(model="qwen/qwen3-vl-8b-instruct")

result = client.generate(
    context="Leia o recibo anexado.",
    images=["recibo.jpg"],  # path, URL, data URI ou bytes
    questions={
        "loja":   {"type": "string", "instructions": "Nome do estabelecimento."},
        "total":  {"type": "number", "instructions": "Valor total, sem símbolo de moeda."},
        "itens":  {"type": "integer", "instructions": "Quantidade de itens."},
        "pago":   {"type": "boolean", "instructions": "Está marcado como pago?"},
        "gorjeta": {"type": ["number", "null"], "instructions": "Gorjeta, se houver.",
                    "depends_on": ["pago"]},
    },
)
# {'loja': 'Sushi Yasuda', 'total': 285.8, 'itens': 3, 'pago': True, 'gorjeta': None}
```

### Tipos suportados

| Tipo | Exemplo |
|---|---|
| `string` | nomes, datas, documentos |
| `integer` | contagens, números inteiros |
| `number` | valores monetários, decimais |
| `boolean` | flags verdadeiro/falso |
| `["tipo", "null"]` | campo opcional (retorna `None` se ausente) |
| `enum` | `{"type": "string", "enum": ["a", "b"]}` (máx. 24 valores) |
| `depends_on` | campo roda depois das dependências, vendo seus resultados |

### Inferência de schema

Se você não sabe os campos de antemão, o modelo pode inferi-los da imagem:

```python
questions = client.infer_questions("Leia a nota fiscal anexada.", images=["nf.jpg"])
result = client.generate("Extraia os campos.", questions=questions, images=["nf.jpg"])
```

## Demo

```bash
python main.py examples/nota-fiscal-do-pedido.jpg
```

Saída:

```
cnpj            string    "09.060.964/0018-48"
clientes        integer   1
impostos_aprox  number    5.69
...
```

## Configuração

| Variável/parâmetro | Default | Descrição |
|---|---|---|
| `OPENROUTER_API_KEY` | — | Chave da API (obrigatória) |
| `model` | `qwen/qwen3-vl-8b-instruct` | Qualquer modelo do OpenRouter; use um com visão para `images=` |
| `temperature` | `0.0` | Temperatura da geração |
| `base_url` | `https://openrouter.ai/api/v1` | Funciona com qualquer API OpenAI-compatible que suporte json_schema strict |

> **Nota:** o modelo precisa suportar structured outputs. Se receber erro 400, tente `google/gemini-2.5-flash`.
