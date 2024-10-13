from typing import Optional
from pydantic import BaseModel, Field

class Football(BaseModel):
    name: Optional[str] = Field(default=None, description="The full name of the football club")
    location: Optional[str] = Field(default=None, description="City and country where the club is based")
    best_player: Optional[int] = Field(default=None, description="The best player of the club")
    founding_year: Optional[int] = Field(default=None, description="Year the club was established")
    valuation: Optional[float] = Field(default=None, description="Club's valuation in billions of dollars")