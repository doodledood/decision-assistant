from typing import List, Optional, Literal, Union

from pydantic import BaseModel, Field


class IndividualParticipantToAddSchema(BaseModel):
    type: Literal['individual']
    name: str = Field(
        description='Name of the participant to add. Be original and select a name that fits the role and mission.')
    role: str = Field(description='Role of the participant to add. Title like "CEO" or "CTO", for example.')
    mission: str = Field(description='Personal mission of the participant to add. Should be a detailed '
                                     'mission statement.')
    symbol: str = Field(description='A unicode symbol of the participant to add (for presentation in chat).')

    def __str__(self):
        return f'{self.symbol} {self.name} ({self.role})'


class TeamParticipantToAddSchema(BaseModel):
    type: Literal['team']
    name: str = Field(
        description='Name of the team to add.')
    mission: str = Field(description='Mission of the team to add. Should be a detailed mission statement.')
    composition_suggestion: str = Field(
        description='List of roles of individual participants or names of sub-teams that are suggested to achieve the '
                    'sub-mission set.')
    symbol: str = Field(description='A unicode symbol of the team to add (for presentation in chat).')

    def __str__(self):
        return f'{self.symbol} {self.name}'


class ManageParticipantsOutputSchema(BaseModel):
    participants_to_remove: List[str] = Field(description='List of participants to be removed.')
    participants_to_add: List[Union[IndividualParticipantToAddSchema, TeamParticipantToAddSchema]] = Field(
        description='List of participants (individuals and teams) to be added. DO NOT include individual participants '
                    'that are a part of a sub-team. The sub-team will handle the composition based on the suggestion. '
                    'For example, if the sub-team "Development Team" is suggested to be added, do not include '
                    'individual participants within that team even if they are mentioned. Instead, suggest the team '
                    'composition within the team definition.', examples=[
            '[{"type": "individual", "name": "John Doe", "role": "CEO", "mission": "Lead the company.", "symbol": "🤵"},'
            '{"type": "team", "name": "Development Team", "mission": "Develop the product.", '
            '"composition_suggestion": "The team should include a Team Leader, Software Engineer, QA Engineer, '
            'and a Software Architect", "symbol": "🛠️"}]'
        ])
    updated_speaker_interaction_schema: Optional[str] = Field(
        description='An updated version of the original interaction schema to better reflect how to achieve the goal.')
    updated_termination_condition: Optional[str] = Field(
        description='An updated version of the termination condition to better reflect the achievement of the goal.')
