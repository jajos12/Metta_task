import google.generativeai as genai
import json
import re
import logging
from typing import Any, Dict, List, Tuple, Set

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

STOP_VERBS = {"find", "show", "give", "get", "compute", "fetch", "tell", "list"}

CALL_REGEX = re.compile(r'!?\(shortestPathFinder\s+([A-Za-z_]+)\s+([A-Za-z_]+)\s+(\d)(?:\s+2)?\)')

def _sanitize_llm_call(raw: str) -> str:
    if not raw:
        return ''
    txt = raw.strip()
    # remove code fences
    txt = re.sub(r'^```[a-zA-Z]*', '', txt)
    txt = txt.replace('```', '').strip()
    # keep only first line containing shortestPathFinder
    for line in txt.splitlines():
        if 'shortestPathFinder' in line:
            txt = line.strip()
            break
    # ensure leading '!'
    if txt.startswith('('):
        txt = '!'+txt
    # strip any leading junk before '!'
    m = re.search(r'!\(shortestPathFinder.*', txt)
    if m:
        txt = m.group(0)
    return txt

def user_query_to_metta_call(user_query: str, gemini_api_key: str) -> Tuple[str, str]:
    """Convert natural language user query into a Metta function call.
    Falls back to a heuristic if LLM response is empty."""
    prompt = f"""Return ONLY a single valid Metta call. No commentary.
Supported form: !(shortestPathFinder <CityFrom> <CityTo> <MetricId> [2])
MetricId: 0=duration 1=cost 2=distance 3=overall. If MetricId=3 add a trailing 2 argument.
Rules:
 - Multi-word city names MUST use underscores.
 - No commentary, backticks, quotes, or explanation.
 - Always start with '!('

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
        raw = (response.text or '').strip()
        logger.info("LLM metta-call raw response: %s", raw[:120])
        txt = _sanitize_llm_call(raw)
        txt2 = txt.replace("shortestPathFinder", "pathFind") if txt else ''
    except Exception as e:  # noqa: BLE001
        logger.error("LLM metta-call generation failed: %s", e)
        txt = ''
        txt2 = ''
    if not CALL_REGEX.match(txt):
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
            if cleaned and cleaned[0].isupper() and cleaned.lower() not in STOP_VERBS:
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
    logger.debug("Sanitized primary metta call: %s", txt)
    logger.debug("Secondary (all paths) call: %s", txt2)
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
EDGE_PATTERN = re.compile(r'\((\w+)\s+--\s+(\d+)\s+--?>\s+(\w+)\)')

def _parse_metta_paths(raw: Any) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if not raw:
        return [], []
    text = str(raw)
    nodes: Set[str] = set()
    edges: List[Dict[str, str]] = []
    for m in EDGE_PATTERN.finditer(text):
        src, weight, dst = m.groups()
        nodes.add(src)
        nodes.add(dst)
        edges.append({"from": src, "to": dst, "label": weight})
    node_objs = [{"id": n} for n in sorted(nodes)]
    return node_objs, edges

def metta_result_to_graph_json(metta_result: str, gemini_api_key: str) -> dict:
    """Deterministically parse Metta path result into graph JSON; fallback to LLM if no edges found."""
    nodes, edges = _parse_metta_paths(metta_result)
    if edges:
        logger.info("Deterministic graph parse: nodes=%d edges=%d", len(nodes), len(edges))
        return {"nodes": nodes, "edges": edges}
    # fallback to LLM minimal (kept but rarely used now)
    prompt = f"Return JSON with nodes (array of strings) and edges (array of objects with from,to,label) for: {metta_result}"[:4000]
    try:
        model = _get_model(gemini_api_key)
        resp = model.generate_content(prompt)
        raw = (resp.text or '').strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            n = data.get('nodes', [])
            # normalize node structure
            if n and isinstance(n[0], str):
                data['nodes'] = [{"id": s} for s in n]
            logger.info("LLM fallback graph nodes=%s edges=%s", len(data.get('nodes', [])), len(data.get('edges', [])))
            return data
    except Exception as e:  # noqa: BLE001
        logger.error("Graph fallback generation failed: %s", e)
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

