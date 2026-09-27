"""Cliente OpenRouter com geração tipada via json_schema strict."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from openai import OpenAI

from .schema import SchemaError, compile_questions, subschema

_MIMES = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp", "gif": "gif"}


def _image_content(image: Any) -> dict:
    """Aceita path local, URL http(s), data URI ou bytes."""
    if isinstance(image, bytes):
        b64 = base64.b64encode(image).decode()
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
    if isinstance(image, (str, Path)):
        image = str(image)
        if image.startswith(("http://", "https://", "data:")):
            return {"type": "image_url", "image_url": {"url": image}}
        ext = image.rsplit(".", 1)[-1].lower()
        mime = _MIMES.get(ext, "jpeg")
        b64 = base64.b64encode(Path(image).read_bytes()).decode()
        return {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}}
    raise TypeError(f"imagem não suportada: {type(image)!r}")


class TypeLLMClient:
    """Geração tipada estilo TypeLLM usando modelos do OpenRouter.

    Os tipos declarados em `questions` são forçados via structured outputs
    (json_schema strict) — o equivalente ao constrained decoding do SGLang.
    """

    def __init__(
        self,
        model: str = "qwen/qwen3-vl-8b-instruct",
        *,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key or os.environ["OPENROUTER_API_KEY"],
            timeout=timeout,
        )

    def _chat(self, context: str, images: Sequence[Any], schema: dict) -> dict:
        content: list[dict] = [_image_content(img) for img in images]
        content.append({"type": "text", "text": context})
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=self.temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "result", "strict": True, "schema": schema},
            },
        )
        return json.loads(response.choices[0].message.content)

    def generate(
        self,
        context: str,
        questions: Mapping[str, Any],
        images: Sequence[Any] = (),
    ) -> dict:
        """Extrai os campos declarados em `questions` com tipos garantidos.

        Campos com `depends_on` rodam em camadas: cada camada vê os resultados
        das anteriores no contexto (como o dependency graph do TypeLLM).
        """
        schema, layers = compile_questions(questions)
        results: dict[str, Any] = {}
        for layer in layers:
            layer_context = context
            if results:
                layer_context += (
                    "\n\nResultados já extraídos:\n"
                    + json.dumps(results, ensure_ascii=False, indent=2)
                )
            results.update(self._chat(layer_context, images, subschema(schema, layer)))
        return results

    def infer_questions(
        self,
        context: str,
        images: Sequence[Any] = (),
    ) -> dict[str, Any]:
        """Infere um dict de `questions` a partir do conteúdo (ex: uma imagem).

        O modelo declara nome, tipo e instruções de cada campo — o passo de
        "schema inference" que o TypeLLM original deixa a cargo do usuário.
        """
        result = self._chat(
            context
            + "\n\nResponda SOMENTE com um JSON Schema válido no formato "
            '{"type": "object", "properties": {...}} com um campo para cada '
            "dado relevante. Para cada propriedade use \"type\" escolhido pela "
            "NATUREZA do dado:\n"
            "- integer: contagens e números inteiros\n"
            "- number: valores monetários e decimais\n"
            "- boolean: flags verdadeiro/falso\n"
            "- string: textos (nomes, datas, documentos)\n"
            'Adicione "null" ao array de type quando o campo pode estar '
            'ausente, e "description" descrevendo o que extrair. '
            "Não inclua campos de enum.",
            images,
            schema={
                "type": "object",
                "properties": {
                    "schema": {
                        "type": "string",
                        "description": "O JSON Schema completo como string.",
                    }
                },
                "required": ["schema"],
                "additionalProperties": False,
            },
        )
        schema = json.loads(result["schema"])
        properties = schema.get("properties")
        if not isinstance(properties, Mapping) or not properties:
            raise SchemaError(f"JSON Schema sem properties:\n{result['schema']}")
        # Converte description -> instructions, o formato de `questions`.
        return {
            name: {
                **{k: v for k, v in field.items() if k in ("type", "enum")},
                "instructions": field.get("description", f'Valor do campo "{name}".'),
            }
            for name, field in properties.items()
        }
