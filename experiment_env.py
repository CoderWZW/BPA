import json
from typing import Dict, Any

from personas_agents import load_personas_and_agents
from group_assignment import (
    split_groups_by_dorm,
    generate_rating_assignments,
    build_anonymous_mapping,
    build_anonymized_assignments,
)
from config import (
    EXPERIMENT_CONFIG,
    PAYOFF_MATRICES,
    AGENT_GROUP_LABEL,
    HUMAN_GROUP_LABEL,
)


def init_experiment() -> Dict[str, Any]:
    persona_path = EXPERIMENT_CONFIG["persona"]["json_path"]
    personas, id2persona, agents, agents_by_id = load_personas_and_agents(persona_path)

    groupA_ids, groupB_ids, groupA_agents, groupB_agents = split_groups_by_dorm(
        personas, agents_by_id, seed=2025
    )

    n_per_type = len(PAYOFF_MATRICES)
    assignments, target_load = generate_rating_assignments(
        groupA_ids=groupA_ids,
        groupB_ids=groupB_ids,
        n_per_type=n_per_type,
        seed=2025,
    )

    number_to_id, id_to_number = build_anonymous_mapping(
        groupA_ids=groupA_ids,
        groupB_ids=groupB_ids,
        seed=2222,
    )

    for pid, ag in agents_by_id.items():
        setattr(ag, "number", id_to_number[pid])

    anonymized_assignments = build_anonymized_assignments(
        assignments=assignments,
        id_to_number=id_to_number,
        seed=4040,
    )

    number_to_group: Dict[int, str] = {}
    for num, pid in number_to_id.items():
        grp = getattr(agents_by_id[pid], "group")
        number_to_group[num] = grp

    experiment_type = EXPERIMENT_CONFIG.get("experiment_type", "agent_vs_agent")

    if experiment_type == "agent_vs_human":
        agent_raters = [
            num for num, grp in number_to_group.items() if grp == AGENT_GROUP_LABEL
        ]
    else:
        agent_raters = sorted(number_to_group.keys())

    with open("id_mapping.json", "w", encoding="utf-8") as f:
        json.dump(
            {"number_to_id": number_to_id, "id_to_number": id_to_number},
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open("anonymized_assignments.json", "w", encoding="utf-8") as f:
        json.dump(
            anonymized_assignments,
            f,
            ensure_ascii=False,
            indent=2,
        )

    env: Dict[str, Any] = {
        "personas": personas,
        "id2persona": id2persona,
        "agents": agents,
        "agents_by_id": agents_by_id,
        "groupA_ids": groupA_ids,
        "groupB_ids": groupB_ids,
        "assignments": assignments,
        "target_load": target_load,
        "number_to_id": number_to_id,
        "id_to_number": id_to_number,
        "anonymized_assignments": anonymized_assignments,
        "number_to_group": number_to_group,
        "experiment_type": experiment_type,
        "agent_raters": agent_raters,
    }
    return env
