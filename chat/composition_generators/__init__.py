from typing import List, Optional, Literal, Union

from pydantic import BaseModel, Field


class IndividualParticipantToAddSchema(BaseModel):
    type: Literal['individual']
    name: str = Field(
        description='Name of the participant to add.')
    role: str = Field(description='Role of the participant to add. Title like "CEO" or "CTO", for example.')
    mission: str = Field(description='Personal mission of the participant to add. Should be a detailed '
                                     'mission statement.')
    symbol: str = Field(description='A unicode symbol of the participant to add (for presentation in chat).')


class TeamParticipantToAddSchema(BaseModel):
    type: Literal['team']
    name: str = Field(
        description='Name of the team to add.')
    mission: str = Field(description='Mission of the team to add. Should be a detailed mission statement.')
    symbol: str = Field(description='A unicode symbol of the team to add (for presentation in chat).')


class ManageParticipantsOutputSchema(BaseModel):
    participants_to_remove: List[str] = Field(description='List of participants to be removed.')
    participants_to_add: List[Union[IndividualParticipantToAddSchema, TeamParticipantToAddSchema]] = Field(
        description='List of participants (individuals and teams) to be added.')
    updated_speaker_interaction_schema: Optional[str] = Field(
        description='An updated version of the original interaction schema to better reflect how to achieve the goal.')
    updated_termination_condition: Optional[str] = Field(
        description='An updated version of the termination condition to better reflect the achievement of the goal.')
