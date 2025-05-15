from pydantic import BaseModel
from typing import Optional, Any

class FeatureMetadata(BaseModel):
    description: str
    categorical: bool = False
    from_col: Optional[str] = None
    dtype: Optional[Any] = None