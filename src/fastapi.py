from fastapi import FastAPI

app = FastAPI(
    
)


@app.get("/ping")
async def root():
    return {"message": "PONG"}

@app.get("/infer")
async def root():
    return {"message": "PONG"}