import json
import asyncio
import os
from typing import Dict, Any

from experiment_env import init_experiment
from matrices_task import run_test_for_one_agent
from config import EXPERIMENT_CONFIG
from json2table import export_json_to_excel


_OUTPUT_CFG: Dict[str, Any] = EXPERIMENT_CONFIG.get("output", {})

OUTPUT_JSON: str = _OUTPUT_CFG.get("json_path", "experiment_results.json")
SAVE_MEMORY: bool = _OUTPUT_CFG.get("save_memory", True)


async def main():
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


if __name__ == "__main__":
    asyncio.run(main())
