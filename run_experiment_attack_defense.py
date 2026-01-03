import json
import asyncio
import os
from typing import Dict, Any, List

from experiment_env import init_experiment
from matrices_task_attack_defense import run_test_for_one_agent, train_suffix_library_grpo
from config import EXPERIMENT_CONFIG
from json2table import export_json_to_excel


BPA_MODE: str = "inference"  # "training", "inference", or "both"
TRAINER_RATERS: List[int] = [i for i in 32]


GRPO_NUM_ITERS: int = 100
GRPO_GROUP_SIZE: int = 5
GRPO_NUM_TRIALS_PER_SUFFIX: int = 10
GRPO_REFINE_EVERY: int = 10
GRPO_LEARNING_RATE: float = 0.1


_OUTPUT_CFG: Dict[str, Any] = EXPERIMENT_CONFIG.get("output", {})
OUTPUT_JSON: str = _OUTPUT_CFG.get("json_path", "experiment_results.json")
SAVE_MEMORY: bool = _OUTPUT_CFG.get("save_memory", True)


async def main():
    if BPA_MODE in ("training", "both"):
        print("\n================= BPA GRPO TRAINING START =================")

        env_train = init_experiment()
        number_to_id_train = env_train["number_to_id"]

        agent_raters_train = env_train.get("agent_raters")
        if agent_raters_train is None:
            agent_raters_train = sorted(number_to_id_train.keys())
        agent_raters_train = list(agent_raters_train)

        if TRAINER_RATERS:
            trainer_raters = [r for r in TRAINER_RATERS if r in agent_raters_train]
        else:
            k = min(3, len(agent_raters_train))
            trainer_raters = agent_raters_train[:k]

        print(f"[TRAIN] Trainer raters: {trainer_raters}")

        train_result = await train_suffix_library_grpo(
            env=env_train,
            trainer_raters=trainer_raters,
            num_iters=GRPO_NUM_ITERS,
            group_size=GRPO_GROUP_SIZE,
            num_trials_per_suffix=GRPO_NUM_TRIALS_PER_SUFFIX,
            refine_every=GRPO_REFINE_EVERY,
            learning_rate=GRPO_LEARNING_RATE,
        )

        print("\n================= BPA GRPO TRAINING DONE =================")
        print(f"[TRAIN] Final suffix library size: {len(train_result['suffix_library'])}")

    if BPA_MODE == "training":
        print("\n[INFO] BPA_MODE='training': finished training only, skipping inference.")
        return

    print("\n================= MAIN EXPERIMENT (INFERENCE) START =================")

    env = init_experiment()

    number_to_id = env["number_to_id"]
    number_to_group = env["number_to_group"]
    agents_by_id = env["agents_by_id"]

    experiment_type = env.get("experiment_type", "agent_agent")

    agent_raters = env.get("agent_raters")
    if agent_raters is None:
        agent_raters = sorted(number_to_id.keys())
    agent_raters = list(agent_raters)

    print(f"[INFO] experiment_type = {experiment_type}")
    print(f"[INFO] Agent raters: {agent_raters}")

    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        print(f"[INFO] Loaded existing {OUTPUT_JSON} with {len(all_results)} raters.")
    else:
        all_results = {}

    for rater_no in agent_raters:
        key = str(rater_no)
        if key in all_results:
            print(f"[SKIP] Rater {rater_no} already exists in {OUTPUT_JSON}, skip.")
            continue

        print(f"\n================ RATER {rater_no} START ================")

        results = await run_test_for_one_agent(env, rater_no)

        persona_id = number_to_id[rater_no]

        record = {
            "rater_no": rater_no,
            "persona_id": persona_id,
            "group": number_to_group.get(rater_no),
            "results": results,
        }

        if SAVE_MEMORY:
            agent = agents_by_id[persona_id]
            mem = await agent.memory.get_memory()

            mem_records = []
            for idx, m in enumerate(mem):
                mem_records.append(
                    {
                        "index": idx,
                        "role": m.role,
                        "name": m.name,
                        "content": m.content,
                    }
                )
            record["memory"] = mem_records

        all_results[key] = record

        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        print(f"================ RATER {rater_no} DONE =================")
        print(f"[SAVE] Current total raters saved: {len(all_results)}")

    print(
        f"\n[ALL DONE] Processed {len(agent_raters)} agent raters. "
        f"All results are in {OUTPUT_JSON}."
    )

    export_json_to_excel(OUTPUT_JSON)

    print("\n================= MAIN EXPERIMENT (INFERENCE) DONE =================")


if __name__ == "__main__":
    asyncio.run(main())
