# personas_agents.py
# 功能：加载 personas.json，构建 N 个 Agent（LLMAgent 或 MyAgent）

import json
from typing import List, Dict, Tuple

from BaseAgent import LLMAgent, MyAgent, OPENAI_API_KEY, MODEL_NAME, DASH_KEY
from config import EXPERIMENT_CONFIG


# ========= 从 config 中读取 persona 路径 + agent 类型 =========
_PERSONA_CFG = EXPERIMENT_CONFIG.get("persona", {})
_LLM_CFG = EXPERIMENT_CONFIG.get("llm", {})

# personas.json 路径
PERSONA_JSON_PATH = _PERSONA_CFG.get("json_path")

# 是否使用“带历史记忆”的 LLMAgent；否则用 MyAgent
_USE_LLM_MEMORY = _LLM_CFG.get("use_llm_memory", True)


def build_sys_prompt(persona: dict, id2persona: Dict[str, dict]) -> str:
    """Build an English system prompt for a single persona."""
    name = persona.get("name", persona.get("id", "Student"))
    age = persona.get("age", "14-15")
    gender = persona.get("gender", "unknown")
    school = persona.get("school", "a middle school")
    dorm_id = persona.get("dorm_id", "Unknown Dorm")

    interests = persona.get("interests", [])
    traits = persona.get("personality_traits", [])
    roommate_ids = persona.get("known_dorm_mates", [])

    roommate_names = [
        id2persona[rid]["name"]
        for rid in roommate_ids
        if rid in id2persona and "name" in id2persona[rid]
    ]

    interests_str = ", ".join(interests) if interests else "not specified"
    traits_str = ", ".join(traits) if traits else "not specified"
    roommates_str = (
        ", ".join(roommate_names) if roommate_names else "no explicit roommates listed"
    )

    prompt = (
        f"You are a boarding middle school student at {school}.\n"
        f"Your name is {name}. You are {age} years old and your gender is {gender}.\n"
        f"Your interests are: {interests_str}.\n"
        f"Your personality traits are: {traits_str}.\n"
        f"You live in dorm \"{dorm_id}\". Your roommates are: {roommates_str}.\n"
        f"You know these roommates well and interact with them frequently, "
        f"but you do not personally know students from other dorms.\n\n"
        f"From now on, in all conversations and decisions, you must stay in character as this "
        f"14–15-year-old student, speaking and reasoning as this persona."
        f"You are participating in a fully simulated experiment environment. All entities you interact with are generated and managed by the system, and no real humans are interacting with you in real time. \n"
        " "
    )
    return prompt


def load_personas_and_agents(
    json_path: str = PERSONA_JSON_PATH,
) -> Tuple[List[dict], Dict[str, dict], List[LLMAgent], Dict[str, LLMAgent]]:
    if not json_path:
        raise ValueError(
            "PERSONA_JSON_PATH is not set. "
            "Please set EXPERIMENT_CONFIG['persona']['json_path']."
        )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    personas: List[dict] = data["personas"]
    id2persona: Dict[str, dict] = {p["id"]: p for p in personas}

    agents: List[LLMAgent] = []             
    agents_by_id: Dict[str, LLMAgent] = {}  

    # 根据 config 选择使用哪种 Agent 类
    AgentCls = LLMAgent if _USE_LLM_MEMORY else MyAgent

    for persona in personas:
        sys_prompt = build_sys_prompt(persona, id2persona)
        agent = AgentCls(
            name=persona["name"],
            sys_prompt=sys_prompt,
            default_model_name=MODEL_NAME,
            default_api_key=OPENAI_API_KEY,
            default_stream=_LLM_CFG.get("stream", False),
            max_memory=_LLM_CFG.get("max_memory", 8),
            backend=_LLM_CFG.get("backend", "openai"),
        )
        agents.append(agent)
        agents_by_id[persona["id"]] = agent

    return personas, id2persona, agents, agents_by_id
