from pydantic import BaseModel, Field

class TrainParamsInput(BaseModel):
    seq_length: int = Field(default=180, description="Number of past days to use as input window")
    hidden_size: int = Field(default=50, description="Number of units in the LSTM layer")
    num_layers: int = Field(default=2, description="Number of stacked LSTM layers")
    learning_rate: float = Field(default=0.005, description="Learning rate (Adam optimizer)")
    num_epochs: int = Field(default=150, description="Number of training epochs")
    train_split: float = Field(default=0.80, description="Fraction of data used for training")