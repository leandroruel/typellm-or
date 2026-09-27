"""Compila o dict de `questions` (estilo TypeLLM) em um JSON Schema strict."""

from __future__ import annotations

from typing import Any, Mapping


class SchemaError(ValueError):
    pass


_VALUE_TYPES = {"string", "integer", "number", "boolean"}


def _compile_field(name: str, field: Mapping[str, Any]) -> tuple[dict, tuple[str, ...]]:
    if not isinstance(field, Mapping):
        raise SchemaError(f"question {name!r} deve ser um objeto")

    field_type = field.get("type")
    nullable = False
    if isinstance(field_type, list):
        kinds = [k for k in field_type if k != "null"]
        if len(field_type) != 2 or len(kinds) != 1:
            raise SchemaError(f'type de {name!r} deve ser um tipo ou [tipo, "null"]')
        field_type, nullable = kinds[0], True
    if field_type not in _VALUE_TYPES:
        raise SchemaError(
            f"type de {name!r} deve ser um de {sorted(_VALUE_TYPES)} (ou [tipo, \"null\"])"
        )

    prop: dict[str, Any] = {"type": [field_type, "null"] if nullable else field_type}

    instructions = field.get("instructions") or field.get("description")
    if instructions is not None and not isinstance(instructions, str):
        raise SchemaError(f"instructions de {name!r} deve ser string")
    prop["description"] = instructions or f'Valor do campo "{name}".'

    enum = field.get("enum")
    if enum is not None:
        if not isinstance(enum, list) or not enum:
            raise SchemaError(f"enum de {name!r} deve ser uma lista não-vazia")
        if len(enum) > 24:
            raise SchemaError(f"enum de {name!r} tem mais de 24 valores")
        prop["enum"] = enum

    depends_on = field.get("depends_on")
    if depends_on is not None:
        if not isinstance(depends_on, list) or any(not isinstance(d, str) for d in depends_on):
            raise SchemaError(f"depends_on de {name!r} deve ser uma lista de nomes")
        depends_on = tuple(depends_on)

    return prop, tuple(depends_on or ())


def compile_questions(questions: Mapping[str, Any]) -> tuple[dict, list[list[str]]]:
    """Retorna (json_schema, camadas) onde camadas agrupa campos por depends_on."""
    if not isinstance(questions, Mapping) or not questions:
        raise SchemaError("questions deve ser um objeto não-vazio")

    properties: dict[str, Any] = {}
    deps: dict[str, tuple[str, ...]] = {}
    for name, field in questions.items():
        if not isinstance(name, str) or not name:
            raise SchemaError("nomes de campo devem ser strings não-vazias")
        properties[name], deps[name] = _compile_field(name, field)

    # Camadas topológicas estáveis (como dependency_layers do TypeLLM).
    for name, dep_names in deps.items():
        for dep in dep_names:
            if dep not in deps:
                raise SchemaError(f"dependência desconhecida {dep!r} em {name!r}")
            if dep == name:
                raise SchemaError(f"campo {name!r} não pode depender de si mesmo")
    layers: list[list[str]] = []
    remaining = set(deps)
    done: set[str] = set()
    while remaining:
        layer = [n for n in questions if n in remaining and set(deps[n]) <= done]
        if not layer:
            raise SchemaError(f"ciclo de dependências entre: {sorted(remaining)}")
        layers.append(layer)
        done.update(layer)
        remaining -= done

    schema = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    return schema, layers


def subschema(schema: dict, names: list[str]) -> dict:
    """Recorta um schema compilado para apenas os campos de uma camada."""
    return {
        "type": "object",
        "properties": {n: schema["properties"][n] for n in names},
        "required": list(names),
        "additionalProperties": False,
    }
