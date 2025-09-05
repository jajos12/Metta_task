import google.generativeai as genai
import json
import re
import logging
from typing import Any, Dict

logger = logging.getLogger("route_app.utils")
logger.setLevel(logging.INFO)

# Gemini model name constant
MODEL_NAME = 'gemini-1.5-flash'

# Utility: Add a new route fact to data.metta and reload Metta knowledge base
def add_route_fact(fact: str, data_file: str = 'data.metta', main_file: str = 'main.metta') -> str:
    fact = fact.strip()
    if not fact.endswith(')'):
        logger.warning("Rejected fact (bad format): %s", fact)
        return 'Invalid fact format.'
    # Append the fact to data.metta
    with open(data_file, 'a', encoding='utf-8') as f:
        f.write('\n' + fact)
    logger.info("Appended new fact to %s: %s", data_file, fact)
    # Optionally, reload Metta knowledge base by re-running main.metta
    from hyperon import metta
    with open(main_file, 'r', encoding='utf-8') as f:
        main_code = f.read()
    metta.run(main_code)
    logger.info("Reloaded Metta knowledge base after fact addition")
    return 'Fact added successfully.'
# Utility: Convert user query to metta call using Gemini
def _get_model(gemini_api_key: str):
    if not gemini_api_key:
        raise ValueError("Missing GEMINI_API_KEY")
    genai.configure(api_key=gemini_api_key)
    return genai.GenerativeModel(MODEL_NAME)

def user_query_to_metta_call(user_query: str, gemini_api_key: str) -> str:
    """Convert natural language user query into a Metta function call.
    Falls back to a heuristic if LLM response is empty."""
    prompt = f"""Return ONLY a single valid Metta call. No commentary.
    Supported form:
        !(shortestPathFinder <CityFrom> <CityTo> <MetricId>)
    MetricId legend: 0=duration 1=cost 2=distance 3=overall

    RULE: Multi-word city names MUST use underscores instead of spaces.
                Example: "Addis Ababa" -> Addis_Ababa, "New York" -> New_York.
    OUTPUT must never contain spaces inside city identifiers.

    Examples:
    Fastest Jimma to Addis => !(shortestPathFinder Jimma Addis 0)
    Cheapest Bahirdar to Mekele => !(shortestPathFinder Bahirdar Mekele 1)
    Shortest distance Hawassa to ArbaMinch => !(shortestPathFinder Hawassa ArbaMinch 2)
    Best overall Addis Ababa to Gonder => !(shortestPathFinder Addis_Ababa Gonder 3)

    User query: {user_query}
    Metta:
    """
    try:
        model = _get_model(gemini_api_key)
        response = model.generate_content(prompt)
        txt = (response.text or '').strip()
        logger.info("LLM metta-call raw response: %s", txt[:120])
    except Exception as e:  # noqa: BLE001
        logger.error("LLM metta-call generation failed: %s", e)
        txt = ''
    if not txt.startswith('!(shortestPathFinder'):
        # heuristic fallback
        metric_map = {
            'fastest': 0, 'duration': 0,
            'cheapest': 1, 'cost': 1,
            'shortest': 2, 'distance': 2,
            'overall': 3, 'best': 3
        }
        lower = user_query.lower()
        metric_id = 3
        for k, v in metric_map.items():
            if k in lower:
                metric_id = v
                break
        # group consecutive capitalized tokens as potential multi-word cities
        raw_tokens = re.split(r'\s+', user_query.strip())
        groups = []
        current = []
        for tok in raw_tokens:
            cleaned = re.sub(r'[^A-Za-z]', '', tok)
            if cleaned and cleaned[0].isupper():
                current.append(cleaned)
            else:
                if current:
                    groups.append('_'.join(current))
                    current = []
        if current:
            groups.append('_'.join(current))
        if len(groups) >= 2:
            city_a, city_b = groups[0], groups[1]
            txt = f"!(shortestPathFinder {city_a} {city_b} {metric_id})"
            txt2 = f"!(pathFinder )"
    logger.debug("Final metta call resolved: %s", txt)
    return txt

# Utility: Convert metta result to human readable using Gemini
def metta_result_to_human(metta_result: str, user_query: str, gemini_api_key: str) -> str:
    """Explain a raw Metta result in natural language."""
    prompt = f"""Explain the following Metta evaluation result briefly and clearly.
If it represents a path, list the cities in order and the optimized metric.

User query: {user_query}
Raw result: {metta_result}
Explanation:
"""
    try:
        model = _get_model(gemini_api_key)
        response = model.generate_content(prompt)
        explanation = (response.text or '').strip()
        if not explanation:
            raise ValueError("Empty explanation text")
        logger.info("LLM explanation generated (len=%d)", len(explanation))
        return explanation
    except Exception as e:  # noqa: BLE001
        logger.error("Explanation generation failed: %s", e)
        return "Unable to generate explanation right now."

# Utility: Convert metta result to graph JSON using Gemini
def metta_result_to_graph_json(metta_result: str, gemini_api_key: str) -> dict:
    """Produce graph JSON from a Metta path result. LLM-assisted, with fallback."""
    prompt = f"""Return ONLY JSON with keys: nodes (list of strings), edges (list of objects {{from,to,label}}).
Infer labels like "Duration: X" / "Cost: Y" etc if present; otherwise just use "route".
Result: {metta_result}
JSON:
"""
    raw = ''
    try:
        model = _get_model(gemini_api_key)
        response = model.generate_content(prompt)
        raw = (response.text or '').strip()
        logger.info("LLM graph response raw: %s", raw[:160])
    except Exception as e:  # noqa: BLE001
        logger.error("Graph JSON generation failed: %s", e)
        raw = ''
    # Extract first JSON object
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            data = json.loads(candidate)
            logger.info("Parsed graph JSON: nodes=%s edges=%s", len(data.get('nodes', [])), len(data.get('edges', [])))
            return data
        except json.JSONDecodeError as je:
            logger.warning("Failed to parse JSON candidate: %s", je)
    # Fallback minimal structure
    logger.info("Returning empty graph structure (fallback)")
    return {"nodes": [], "edges": []}


def looks_like_path(result: Any) -> bool:
    """Heuristic to decide if Metta result likely encodes path(s)."""
    if result is None:
        return False
    if isinstance(result, str):
        tokens = result.lower()
        return ('--' in tokens) or ('flight-route' in tokens)
    if isinstance(result, (list, tuple)):
        # If any element is string containing '--'
        return any(isinstance(x, str) and '--' in x for x in result)
    return False

