"""Week 4 - Output schema for the clinical safety report (FR-4.1, FR-4.5)."""
from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator

from constants import DISCLAIMER


class ClinicalSafetyReport(BaseModel):
    """Exactly the five fields required by the PRD. Unknown extra fields are rejected."""

    model_config = ConfigDict(extra="forbid")

    Eligible_For_Vaccine: bool = Field(
        ..., description="Whether the patient is eligible given risk and guidelines")
    Recommended_Dose_mg: float = Field(
        ..., description="Recommended dose in MILLIGRAMS, must respect retrieved guideline bounds")
    Reasoning_Steps: List[str] = Field(
        ..., description="Ordered, human-readable reasoning steps behind the decision")
    FDA_Safety_Warnings_Cited: List[str] = Field(
        ..., description="Exact guideline snippets the recommendation relies on")
    Disclaimer: str = Field(
        default=DISCLAIMER, description="Must always be present in the final report")

    @field_validator("Recommended_Dose_mg")
    @classmethod
    def _dose_not_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Recommended_Dose_mg cannot be negative")
        return v

    @field_validator("Reasoning_Steps")
    @classmethod
    def _needs_reasoning(cls, v: List[str]) -> List[str]:
        if not v or not any(step.strip() for step in v):
            raise ValueError("Reasoning_Steps must contain at least one non-empty step")
        return v

    @field_validator("Disclaimer")
    @classmethod
    def _disclaimer_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Disclaimer must not be empty")
        return v
