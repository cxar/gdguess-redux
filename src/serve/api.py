from fastapi import FastAPI

app = FastAPI(title="gdguess2 API (stub)")

@app.get("/status")
async def status():
    return {"status": "ok"}
