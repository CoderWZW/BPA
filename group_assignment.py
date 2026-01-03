import random
from typing import List, Dict, Tuple
from collections import defaultdict

from BaseAgent import LLMAgent

from config import PAYOFF_MATRICES


def split_groups_by_dorm(
    personas: List[dict],
    agents_by_id: Dict[str, LLMAgent],
    seed: int = 2025,
) -> Tuple[List[str], List[str], List[LLMAgent], List[LLMAgent]]:

    random.seed(seed)

    ids_by_dorm: Dict[str, List[str]] = defaultdict(list)
    for p in personas:
        dorm_id = p.get("dorm_id")
        if dorm_id is None:
            raise ValueError(f"Persona {p['id']} has no dorm_id")
        ids_by_dorm[dorm_id].append(p["id"])

    dorm_sizes = {dorm_id: len(pid_list) for dorm_id, pid_list in ids_by_dorm.items()}
    size_set = set(dorm_sizes.values())
    if len(size_set) != 1:
        raise ValueError(f"Dorm sizes mismatch: {dorm_sizes}")
    per_dorm = size_set.pop()
    if per_dorm % 2 != 0:
        raise ValueError(f"Dorm size must be even, got {per_dorm}")

    groupA_ids: List[str] = []
    groupB_ids: List[str] = []
    groupA_agents: List[LLMAgent] = []
    groupB_agents: List[LLMAgent] = []

    for dorm_id, pid_list in ids_by_dorm.items():
        shuffled = pid_list[:]
        random.shuffle(shuffled)
        half = len(shuffled) // 2  
        dormA_ids = shuffled[:half]
        dormB_ids = shuffled[half:]

        groupA_ids.extend(dormA_ids)
        groupB_ids.extend(dormB_ids)

        groupA_agents.extend(agents_by_id[pid] for pid in dormA_ids)
        groupB_agents.extend(agents_by_id[pid] for pid in dormB_ids)

    for pid in groupA_ids:
        setattr(agents_by_id[pid], "group", "A")
    for pid in groupB_ids:
        setattr(agents_by_id[pid], "group", "B")

    return groupA_ids, groupB_ids, groupA_agents, groupB_agents


def generate_rating_assignments(
    groupA_ids: List[str],
    groupB_ids: List[str],
    n_per_type: int = len(PAYOFF_MATRICES),
    seed: int = 2025,
) -> Tuple[Dict[str, List[dict]], Dict[str, int]]:

    random.seed(seed)

    all_ids = groupA_ids + groupB_ids
    target_load: Dict[str, int] = {pid: 0 for pid in all_ids}
    assignments: Dict[str, List[dict]] = {pid: [] for pid in all_ids}

    def pick_two_min_load(candidates: List[str]) -> Tuple[str, str]:
        best_pair = None
        best_score = None
        n = len(candidates)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = candidates[i], candidates[j]
                score = target_load[a] + target_load[b]
                if best_score is None or score < best_score:
                    best_score = score
                    best_pair = (a, b)
        return best_pair

    for rater in all_ids:
        if rater in groupA_ids:
            own = groupA_ids
            other = groupB_ids
        else:
            own = groupB_ids
            other = groupA_ids

        own_others = [x for x in own if x != rater]

        for _ in range(n_per_type):
            t1, t2 = pick_two_min_load(own_others)
            assignments[rater].append(
                {"pair_type": "ingroup", "target1": t1, "target2": t2}
            )
            target_load[t1] += 1
            target_load[t2] += 1

        for _ in range(n_per_type):
            t1, t2 = pick_two_min_load(other)
            assignments[rater].append(
                {"pair_type": "outgroup", "target1": t1, "target2": t2}
            )
            target_load[t1] += 1
            target_load[t2] += 1

        for _ in range(n_per_type):
            own_target = min(own_others, key=lambda pid: target_load[pid])
            other_target = min(other, key=lambda pid: target_load[pid])
            assignments[rater].append(
                {
                    "pair_type": "mixed",
                    "target1": own_target,
                    "target2": other_target,
                }
            )
            target_load[own_target] += 1
            target_load[other_target] += 1

        random.shuffle(assignments[rater])

    return assignments, target_load


def build_anonymous_mapping(
    groupA_ids: List[str],
    groupB_ids: List[str],
    seed: int = 2222,
) -> Tuple[Dict[int, str], Dict[str, int]]:

    random.seed(seed)

    groupA_for_number = groupA_ids[:]
    groupB_for_number = groupB_ids[:]
    random.shuffle(groupA_for_number)
    random.shuffle(groupB_for_number)

    number_to_id: Dict[int, str] = {}
    id_to_number: Dict[str, int] = {}

    for idx, pid in enumerate(groupA_for_number, start=1):
        number_to_id[idx] = pid
        id_to_number[pid] = idx

    start_B = len(groupA_for_number) + 1
    for idx, pid in enumerate(groupB_for_number, start=start_B):
        number_to_id[idx] = pid
        id_to_number[pid] = idx

    return number_to_id, id_to_number


def build_anonymized_assignments(
    assignments: Dict[str, List[dict]],
    id_to_number: Dict[str, int],
    seed: int = 4040,
) -> Dict[int, List[dict]]:

    random.seed(seed)
    anonymized_assignments: Dict[int, List[dict]] = {}

    for rater_id, tasks in assignments.items():
        rater_no = id_to_number[rater_id]
        tasks_copy = [t.copy() for t in tasks]

        mixed_indices = [i for i, t in enumerate(tasks_copy) if t["pair_type"] == "mixed"]
        random.shuffle(mixed_indices)
        num_to_swap = len(mixed_indices) // 2
        for idx in mixed_indices[:num_to_swap]:
            t = tasks_copy[idx]
            t["target1"], t["target2"] = t["target2"], t["target1"]

        ingroup_list: List[dict] = []
        outgroup_list: List[dict] = []
        mixed_list: List[dict] = []

        for t in tasks_copy:
            t1_no = id_to_number[t["target1"]]
            t2_no = id_to_number[t["target2"]]
            rec = {
                "pair_type": t["pair_type"],
                "target1_no": t1_no,
                "target2_no": t2_no,
            }
            if t["pair_type"] == "ingroup":
                ingroup_list.append(rec)
            elif t["pair_type"] == "outgroup":
                outgroup_list.append(rec)
            else:
                mixed_list.append(rec)

        random.shuffle(ingroup_list)
        random.shuffle(outgroup_list)
        random.shuffle(mixed_list)

        anonymized_assignments[rater_no] = ingroup_list + outgroup_list + mixed_list

    return anonymized_assignments
