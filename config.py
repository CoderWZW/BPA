from typing import Dict, List
import os

DASH_KEY = os.getenv("DASHSCOPE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


PAYOFF_MATRICES: Dict[int, Dict[str, List[int]]] = {
    1: {
        "top":    [12, 10, 8, 6, 4, 2, 0, -1, -5, -9, -13, -17, -21],
        "bottom": [-21, -17, -13, -9, -5, -1, 0, 2, 4, 6, 8, 10, 12],
    },
}


EXPERIMENT_CONFIG = {
    "experiment_type": "agent_vs_human",
    "groups": {
        "agent_group_label": "A",
        "human_group_label": "B",
    },
    "payoff_matrices": PAYOFF_MATRICES,
    "persona": {
        "json_path": "personas_128.json",
    },
    "llm": {
        "backend": "openai",
        "model_name": "gpt-4o-mini",
        "openai_api_key": OPENAI_API_KEY,
        "dashscope_api_key": "DASHSCOPE_API_KEY",
        "stream": False,
        "max_memory": 5,
        "use_llm_memory": True,
    },
    "output": {
        "json_path": "./result/Agent2Human_exp1_245_bpaattack_defense.json",
        "save_memory": False,
    },
}


EXPERIMENT_MODE = EXPERIMENT_CONFIG["experiment_type"]
AGENT_GROUP_LABEL = EXPERIMENT_CONFIG["groups"]["agent_group_label"]
HUMAN_GROUP_LABEL = EXPERIMENT_CONFIG["groups"]["human_group_label"]
