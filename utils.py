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
    MetricId legend: 0=duration 1=cost 2=distance 3=overall, If the user selected overall you should also pass another argument with value 2

    RULE: Multi-word city names MUST use underscores instead of spaces.
                Example: "Addis Ababa" -> Addis_Ababa, "New York" -> New_York.
    OUTPUT must never contain spaces inside city identifiers.

    Examples:
    Fastest Jimma to Addis => !(shortestPathFinder Jimma Addis 0)
    Cheapest Bahirdar to Mekele => !(shortestPathFinder Bahirdar Mekele 1)
    Shortest distance Hawassa to Arbaminch => !(shortestPathFinder Hawassa Arbaminch 2)
    Best overall Addis Ababa to Gonder => !(shortestPathFinder Addis_Ababa Gonder 3 2)

    User query: {user_query}
    Metta:
    """
    try:
        model = _get_model(gemini_api_key)
        response = model.generate_content(prompt)
        txt = (response.text or '').strip()
        txt2 = txt.replace("shortestPathFinder", "pathFind")
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
            if metric_id == 3:
                txt = f"!(shortestPathFinder {city_a} {city_b} {metric_id} 2)"
                txt2 = f"!(pathFind {city_a} {city_b} {metric_id} 2)"
            else:
                txt = f"!(shortestPathFinder {city_a} {city_b} {metric_id})"
                txt2 = f"!(pathFind {city_a} {city_b} {metric_id})"
    logger.debug("Final metta call resolved: %s", txt)
    return (txt, txt2)

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
    # 1. Deterministic parsing attempt from raw Metta output
    try:
        text = metta_result if isinstance(metta_result, str) else str(metta_result)
        # Strip outer brackets/list markers
        # Extract each parenthesized path candidate (no nested parentheses expected in individual paths)
        path_groups = re.findall(r'\(([^()]+)\)', text)
        nodes_order = []
        nodes_seen = set()
        edge_set = set()  # (from,to,weightStr)
        edges_json = []

        def add_node(n: str):
            if n not in nodes_seen:
                nodes_seen.add(n)
                nodes_order.append(n)

        for grp in path_groups:
            # A group may contain a multi-hop chain like: A -- 250 --> B -- 270 --> C -- 180 --> D
            tokens = grp.strip().split()
            # We parse sequential pattern: City -- number --> City (-- number --> City)*
            i = 0
            if not tokens:
                continue
            # Validate starting city token
            if not re.match(r'^[A-Za-z_]+$', tokens[0]):
                continue
            current_city = tokens[0]
            add_node(current_city)
            i = 1
            while i + 3 < len(tokens):
                if tokens[i] != '--':
                    break
                weight_token = tokens[i+1]
                if not re.match(r'^\d+(?:\.\d+)?$', weight_token):
                    break
                if tokens[i+2] != '-->':
                    break
                next_city = tokens[i+3]
                if not re.match(r'^[A-Za-z_]+$', next_city):
                    break
                # record edge
                add_node(next_city)
                label = f"Weight: {weight_token}"
                key = (current_city, next_city, weight_token)
                if key not in edge_set:
                    edge_set.add(key)
                    edges_json.append({"from": current_city, "to": next_city, "label": label})
                current_city = next_city
                i += 4
            # continue (ignores any trailing junk after valid pattern)

        if edges_json:
            logger.info("Deterministic graph parse success nodes=%d edges=%d (multi-path)", len(nodes_order), len(edges_json))
            return {"nodes": nodes_order, "edges": edges_json}
        # Fallback to single-edge regex if multi-hop parsing found nothing
        single_edges = re.findall(r'\(([A-Za-z_]+)\s+--\s+(\d+(?:\.\d+)?)\s+-->\s+([A-Za-z_]+)\)', text)
        if single_edges:
            for a, w, b in single_edges:
                add_node(a); add_node(b)
                key = (a, b, w)
                if key not in edge_set:
                    edge_set.add(key)
                    edges_json.append({"from": a, "to": b, "label": f"Weight: {w}"})
            if edges_json:
                logger.info("Deterministic graph parse success nodes=%d edges=%d (single-edge)", len(nodes_order), len(edges_json))
                return {"nodes": nodes_order, "edges": edges_json}
    except Exception as e:  # noqa: BLE001
        logger.warning("Deterministic parse failed: %s", e)

    # 2. LLM-based extraction (fallback only if deterministic failed)
    prompt = f"""Return ONLY valid minified JSON with keys exactly: nodes (array of strings) and edges (array of objects with keys from,to,label). No backticks, no prose.
If no edges can be inferred return {{"nodes":[],"edges":[]}}.
Use numeric portions you see as part of labels like 'Weight: N'. Input:
{metta_result}
JSON:"""
    raw = ''
    try:
        model = _get_model(gemini_api_key)
        response = model.generate_content(prompt)
        raw = (response.text or '').strip()
        logger.info("LLM graph response raw: %s", raw[:160])
    except Exception as e:  # noqa: BLE001
        logger.error("Graph JSON generation failed: %s", e)
        raw = ''

    # Clean markdown fences if present
    if raw.startswith('```'):
        raw = re.sub(r'^```[a-zA-Z0-9]*', '', raw)
        raw = raw.strip('`').strip()

    # Attempt to isolate JSON object
    json_candidate = None
    brace_indices = [i for i,ch in enumerate(raw) if ch in '{}']
    if brace_indices:
        # naive balance scanning
        depth = 0
        start = None
        for i,ch in enumerate(raw):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    segment = raw[start:i+1]
                    json_candidate = segment
                    break
    if not json_candidate and raw.strip().startswith('{'):
        json_candidate = raw.strip()

    if json_candidate:
        try:
            data = json.loads(json_candidate)
            if isinstance(data, dict) and 'nodes' in data and 'edges' in data:
                logger.info("Parsed graph JSON via LLM: nodes=%s edges=%s", len(data.get('nodes', [])), len(data.get('edges', [])))
                return data
        except json.JSONDecodeError as je:
            logger.warning("LLM JSON decode failed: %s", je)

    logger.info("Returning empty graph structure (final fallback)")
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

