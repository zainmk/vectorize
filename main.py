from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
import json
import os
import bm25
import uvicorn
from semantic import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    print('server init...')

    global MOVIES
    global tok_data
    global semantic_collection

    with open(os.path.join(BASE_DIR, 'movies.json')) as f:
        MOVIES = json.load(f)
    tok_data = bm25.build_index(MOVIES)
    semantic_collection = build_semantic_engine()
    
    yield

    print('server close...')
    pass

app = FastAPI(lifespan=lifespan)

class SearchRequest(BaseModel):
    query: str


@app.get("/") # FAST API ROUTE DECLARED FOR LOCAL DEV
async def serve_app():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


@app.post("/search")
async def search(req: SearchRequest):
    
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    top = bm25.search(req.query, tok_data, MOVIES)[:5] # TOP 5 RESULTS
    max_score = top[0][0] if top and top[0][0] > 0 else 1

    results = [
        {
            "id": movie["id"],
            "title": movie["title"],
            "year": movie["year"],
            "genre": movie["genre"],
            "description": movie["description"],
            "similarity": round((score / max_score) * 100, 1),
        }
        for score, movie in top
        if score > 0
    ]

    return {"query": req.query, "results": results}


@app.post("/semanticsearch")
async def search(req: SearchRequest):
    
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    res = semanticsearch(req.query, semantic_collection)

    return {"query": req.query, "results": res }


@app.get("/movies")
async def list_movies(): 
    return MOVIES


if __name__ == "__main__": uvicorn.run("main:app", reload=True)