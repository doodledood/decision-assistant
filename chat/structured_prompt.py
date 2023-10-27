class StructuredPrompt(dict):
    def __init__(self, parts: dict):
        super().__init__(parts)

    def format_dict(self, d: dict, level: int = 0) -> str:
        parts = []
        for k, v in d.items():
            result = f'{"#" * (level + 1)} {k.upper() if level == 0 else k.capitalize()}\n'

            if isinstance(v, dict):
                result += self.format_dict(v, level + 1)
            elif isinstance(v, list):
                result += '\n'.join([f'- {item}' for item in v])
            elif isinstance(v, tuple):
                text, other = v
                result += text + '\n\n' + self.format_dict(other, level + 1)
            else:
                result += v

            parts.append(result)

        result = '\n\n'.join(parts)

        return result

    def __str__(self):
        return self.format_dict(d=self)

    def __repr__(self):
        return str(self)
