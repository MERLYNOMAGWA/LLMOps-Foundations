from fastapi import FastAPI
from llm_inference.schemas import GenerationRequest, GenerationResponse
from llm_inference.model_loader import generator

app = FastAPI(title="LLM Inference API")

@app.get("/")
def root():
   return {"message": "LLM Inference API is running!"}

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(req: GenerationRequest):
    result = generator(
        req.prompt,
        max_length=req.max_length,
        temperature=req.temperature,
        num_return_sequences=1
    )
    return {"generated_text": result[0]['generated_text']}
 