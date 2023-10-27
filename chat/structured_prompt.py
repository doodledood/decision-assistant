import dataclasses
from typing import Optional, List, Union


@dataclasses.dataclass
class Section:
    name: str
    content: Optional[Union[str, List[str]]] = None
    sub_sections: Optional[List['Section']] = None

    def to_text(self, level: int = 0) -> str:
        result = f'{"#" * (level + 1)} {self.name.upper() if level == 0 else self.name.capitalize()}'

        if self.content is not None:
            if isinstance(self.content, list):
                result += '\n'.join([f'- {item}' for item in self.content])
            else:
                result += '\n' + self.content

        if self.sub_sections is not None:
            for sub_section in self.sub_sections:
                result += '\n\n' + sub_section.to_text(level + 1)

        return result

    @staticmethod
    def from_dict(name: str, content: Union[str, List[str], dict]) -> 'Section':
        if isinstance(content, dict):
            sub_sections = []

            for sub_section_name, sub_section_content in content.items():
                sub_sections.append(Section.from_dict(sub_section_name, sub_section_content))

            return Section(name=name, sub_sections=sub_sections)
        else:
            return Section(name=name, content=content)


@dataclasses.dataclass
class StructuredPrompt:
    sections: List[Section]

    def to_text(self) -> str:
        result = ''
        for section in self.sections:
            result += section.to_text() + '\n\n'

        return result

    @staticmethod
    def from_dict(d: dict) -> 'StructuredPrompt':
        sections = []

        for name, contents in d.items():
            sections.append(Section.from_dict(name, contents))

        return StructuredPrompt(sections=sections)


if __name__ == '__main__':
    print(StructuredPrompt(
        sections=[
            Section(
                name='Goal Identification',
                content='Identify the goal of the decision you are trying to make. For example, if you are trying to decide which car to buy, your goal might be to buy the car that is the most reliable.',
                sub_sections=[
                    Section(
                        name='Goal',
                        content='Buy the car that is the most reliable.'
                    ),
                    Section(
                        name='Alternatives',
                        content=[
                            'Buy a Honda Civic.',
                            'Buy a Toyota Corolla.',
                            'Buy a Ford Focus.'
                        ]
                    )
                ]
            ),
            Section(
                name='Criteria Identification',
                content='Identify the criteria that you will use to evaluate the alternatives. For example, if you are trying to decide which car to buy, your criteria might be reliability and price.',
                sub_sections=[
                    Section(
                        name='Criteria',
                        content=[
                            'Reliability',
                            'Price'
                        ],
                        sub_sections=[
                            Section(
                                name='Reliability',
                                content='The car that is the most reliable.'
                            )
                        ])
                ]
            )]).to_text())
