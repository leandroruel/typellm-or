import json
import os
import sys

from typellm_or import TypeLLMClient

client = TypeLLMClient(model=os.environ.get("MODEL", "qwen/qwen3-vl-8b-instruct"))


def read_nota_fiscal(image_path: str) -> dict:
    questions = client.infer_questions(
        "Leia a nota fiscal anexada.",
        images=[image_path],
    )
    return client.generate(
        "Leia a nota fiscal anexada e extraia os campos descritos. "
        "Valores monetários como número puro, sem símbolo de moeda. "
        "Use null para campos ausentes ou ilegíveis.",
        questions=questions,
        images=[image_path],
    )


def _type_name(value) -> str:
    return {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }[type(value)]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python main.py examples/nota-fiscal-do-pedido.jpg")
        sys.exit(1)
    data = read_nota_fiscal(sys.argv[1])
    width = max(len(k) for k in data)
    for name, value in data.items():
        print(f"{name.ljust(width)}  {_type_name(value).ljust(8)}  {json.dumps(value, ensure_ascii=False)}")
