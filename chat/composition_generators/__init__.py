from typing import List, Optional

from pydantic import BaseModel, Field


class ParticipantToAddSchema(BaseModel):
    name: str = Field(description='Name of the participant to add. If `team` is also provided, this will be the name '
                                  'of the team instead.')
    role: str = Field(description='Role of the participant to add. Title like "CEO" or "CTO", for example. If a '
                                  '`team` is provided, this will be the name of the team as well.')
    team: Optional[str] = Field(description='Team of the participant to add. For example, "Engineering" or "Sales". '
                                            'If team field is provided, the participant will be the front face of an '
                                            'entire team and their representative in the chat. If not provided, '
                                            'the participant will be a solo specialist participant. Provide a team if '
                                            'you think the mission for the participant is too broad or difficult for '
                                            'a single specialist.')
    mission: str = Field(description='Personal mission of the participant to add. Should be a detailed '
                                     'mission statement. If a `team` is provided, this will be the mission and goal '
                                     'of the entire team.')
    symbol: str = Field(description='A unicode symbol of the participant to add (for presentation in chat).')


class ManageParticipantsOutputSchema(BaseModel):
    participants_to_remove: List[str] = Field(description='List of participants to be removed.')
    participants_to_add: List[ParticipantToAddSchema] = Field(description='List of participants to be added.')
    updated_speaker_interaction_schema: Optional[str] = Field(
        description='An updated version of the original interaction schema to better reflect how to achieve the goal.')
    updated_termination_condition: Optional[str] = Field(
        description='An updated version of the termination condition to better reflect the achievement of the goal.')
