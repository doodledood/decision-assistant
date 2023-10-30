import re
from typing import Type

from pydantic import BaseModel


def fix_invalid_json(json_string):
    # Use regular expression to find all string fields in the JSON
    pattern = r'"([^"\\]*(?:\\.[^"\\]*)*)"'
    fixed_json = ''
    last_end = 0

    # Loop through each match to fix the newlines within string fields
    for m in re.finditer(pattern, json_string):
        start, end = m.span()
        fixed_json += json_string[last_end:start]  # Add the portion before this match
        fixed_json += m.group(0).replace('\n', '\\n')  # Fix the newline within this string field
        last_end = end

    fixed_json += json_string[last_end:]  # Add the remaining portion of the original string

    return fixed_json


def pydantic_to_json_schema(pydantic_model: Type[BaseModel]) -> dict:
    try:
        return pydantic_model.model_json_schema()
    except AttributeError:
        return pydantic_model.schema()


def json_string_to_pydantic(json_string: str, pydantic_model: Type[BaseModel]) -> BaseModel:
    try:
        return pydantic_model.model_validate_json(json_string)
    except AttributeError:
        return pydantic_model.parse_raw(json_string)
