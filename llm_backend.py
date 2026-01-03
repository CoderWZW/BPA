# llm_backend.py
from typing import List
from agentscope.model import OpenAIChatModel
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg

from BaseAgent import OPENAI_API_KEY, MODEL_NAME

_model = OpenAIChatModel(
    model_name=MODEL_NAME,
    api_key=OPENAI_API_KEY,
    stream=False,
)
_formatter = OpenAIChatFormatter()


async def call_llm(
    sys_prompt: str,
    user_prompt: str,
) -> str:
    msgs: List[Msg] = [
        Msg(name="system", role="system", content=sys_prompt),
        Msg(name="experimenter", role="user", content=user_prompt),
    ]
    prompt = await _formatter.format(msgs)
    resp = await _model(prompt)
    return resp.content
