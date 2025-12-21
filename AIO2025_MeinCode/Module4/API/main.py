from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

# Create FastAPI app instance
app = FastAPI(title="My API", version="0.1.0")

# Example Pydantic model
class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Hello from API!"}

# Example POST endpoint
@app.post("/items/")
def create_item(item: Item):
    return {"item": item}

# Preprocessing Doc

# Chunking Doc

# Parsing Chunks

# Upload Parsed Chunks to Qdrant


# Generate Output
# Context = Retrieve Parsed Chunks from Qdrant
# Custom_generate(context, user_query)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
