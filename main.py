from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
from dotenv import load_dotenv
from hyperon import MeTTa
from utils import (
    user_query_to_metta_call,
    metta_result_to_human,
    metta_result_to_graph_json,
    add_route_fact,
)
import logging
import re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)
logger = logging.getLogger("route_app.main")

# Load environment variables from .env file
load_dotenv()

# Use GEMINI_API_KEY for Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI()
metta = MeTTa()


# Load and preprocess data once at startup
with open('main.metta', 'r', encoding='utf-8') as f:
    main_metta_code = f.read()
metta.run(main_metta_code)

# Serve static files (UI)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/route")
async def get_route(request: Request):
    data = await request.json()
    user_query = data.get("query")
    if not user_query:
        return JSONResponse({"error": "Missing query"}, status_code=400)

    # 1. Convert user query to metta call
    if re.match(r"^(hi|hello|hey|howdy|yo|sup|good (morning|evening|afternoon))\b", user_query, re.I):
        return {"answer": "👋 Hi! Ask me for routes like 'Fastest Jimma to Addis_Ababa' or add a fact with (flight-route CityA CityB (Duration d Cost c Distance x))"}

    metta_call, for_graph = user_query_to_metta_call(user_query, GEMINI_API_KEY)
    logger.info("Resolved metta call(all paths): %s", for_graph)
    logger.info("Resolved metta call(shortest path): %s", metta_call)

    # 2. Run metta call
    try:
        result = metta.run(metta_call); 
        result2 = metta.run(for_graph); 
        logger.info("The result returned from metta is(all paths): %s", result2)
        logger.info("The result returned from metta is(shortest path): %s", result)
    except Exception as e:  # noqa: BLE001
        logger.exception("Metta execution failed")
        return JSONResponse({
            "metta_call": f"{metta_call}",
            "error": f"Metta execution error: {e}"}, status_code=500)

    # 3. Convert metta result to human readable
    human_answer = metta_result_to_human(result, user_query, GEMINI_API_KEY)

    # 4. Convert metta result to graph JSON
    graph_json = metta_result_to_graph_json(result2, GEMINI_API_KEY)

    return JSONResponse({
        "metta_call": f"{metta_call}",
        "result": f"{result}",
        "answer": human_answer,
        "graph": graph_json
    })

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/add-fact")
async def add_fact(request: Request):
    data = await request.json()
    fact = data.get("fact", "").strip()
    if not fact:
        return JSONResponse({"error": "Missing fact"}, status_code=400)
    status = add_route_fact(fact)
    logger.info("Add fact status: %s", status)
    return {"status": status}

@app.get("/")
def root():
    return FileResponse("static/index.html")
