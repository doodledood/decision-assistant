import dataclasses
from typing import Optional, List


@dataclasses.dataclass
class Section:
    name: str
    text: Optional[str] = None
    list: Optional[List[str]] = None
    sub_sections: Optional[List['Section']] = None
    list_item_prefix: Optional[str] = '-'

    def to_text(self, level: int = 0) -> str:
        result = f'{"#" * (level + 1)} {self.name.upper() if level == 0 else self.name.capitalize()}'

        if self.text is not None:
            result += '\n' + self.text

        if self.list is not None:
            result += '\n' + '\n'.join(
                [f'{self.list_item_prefix if self.list_item_prefix else str(i + 1) + "."} {item}' for i, item in
                 enumerate(self.list)])

        if self.sub_sections is not None:
            for sub_section in self.sub_sections:
                result += '\n\n' + sub_section.to_text(level + 1)

        return result


@dataclasses.dataclass
class StructuredPrompt:
    sections: List[Section]

    def __getitem__(self, item):
        assert isinstance(item, str)

        relevant_sections = [section for section in self.sections if section.name == item]
        if len(relevant_sections) == 0:
            raise KeyError(f'No section with name {item} exists.')

        return relevant_sections[0]

    def __setitem__(self, key, value):
        assert isinstance(key, str)
        assert isinstance(value, Section)

        try:
            section = self[key]
            section.text = value
        except KeyError:
            self.sections.append(value)

    def __str__(self) -> str:
        result = ''
        for section in self.sections:
            result += section.to_text() + '\n\n'

        return result

    def __repr__(self):
        return self.__str__()


if __name__ == '__main__':
    print(StructuredPrompt(
        sections=[
            Section(
                name='Goal Identification',
                text='Identify the goal of the decision you are trying to make. For example, if you are trying to decide which car to buy, your goal might be to buy the car that is the most reliable.',
                sub_sections=[
                    Section(
                        name='Goal',
                        text='Buy the car that is the most reliable.'
                    ),
                    Section(
                        name='Alternatives',
                        text='Here are some alternatives to consider:',
                        list=[
                            'Buy a Honda Civic.',
                            'Buy a Toyota Corolla.',
                            'Buy a Ford Focus.'
                        ]
                    )
                ]
            ),
            Section(
                name='Criteria Identification',
                text='Identify the criteria that you will use to evaluate the alternatives. For example, if you are trying to decide which car to buy, your criteria might be reliability and price.',
                sub_sections=[
                    Section(
                        name='Criteria',
                        list=[
                            'Reliability',
                            'Price'
                        ],
                        sub_sections=[
                            Section(
                                name='Reliability',
                                text='The car that is the most reliable.'
                            )
                        ])
                ]
            )]))
