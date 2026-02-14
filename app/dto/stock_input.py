from pydantic import BaseModel, Field

class StockInput(BaseModel):
    # Input is a list of lists: [[Open, High, Low, Vol, Close], ...]
    features: list[list[float]] = Field(
        ..., 
        description="List of 60 days. Each day must have 5 values: [Open, High, Low, Volume, Close]",
        example=[
            [30.1, 30.5, 29.8, 10000.0, 30.2], 
            [30.2, 30.8, 30.0, 15000.0, 30.5] 
            # ... (60 rows in total)
        ]
    )