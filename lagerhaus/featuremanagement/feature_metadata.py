from pydantic import BaseModel

class FeatureMetadata(BaseModel):
    description: str
    categorical: bool = False