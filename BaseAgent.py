import os
import re
from typing import Optional, Union, List

from agentscope.memory import InMemoryMemory
from agentscope.agent import AgentBase
from agentscope.model import DashScopeChatModel, OpenAIChatModel
from agentscope.formatter import DashScopeChatFormatter, OpenAIChatFormatter
from agentscope.message import Msg

from config import EXPERIMENT_CONFIG

_LLMCFG = EXPERIMENT_CONFIG.get("llm", {})

DEFAULT_BACKEND = _LLMCFG.get("backend", "openai").lower()
MODEL_NAME = _LLMCFG.get("model_name", "gpt-4o-mini")

_openai_env = _LLMCFG.get("openai_api_key", "OPENAI_API_KEY")
_dash_env = _LLMCFG.get("dashscope_api_key", "DASHSCOPE_API_KEY")

OPENAI_API_KEY = os.getenv(_openai_env)
DASH_KEY = os.getenv(_dash_env)

DEFAULT_STREAM = _LLMCFG.get("stream", False)
DEFAULT_MAX_MEMORY = _LLMCFG.get("max_memory", 8)


class MyAgent(AgentBase):
    def __init__(
        self,
        name: str = "Friday",
        sys_prompt: str = "You are an AI assistant named Friday.",
        model: Optional[Union[OpenAIChatModel, DashScopeChatModel]] = None,
        formatter: Optional[Union[OpenAIChatFormatter, DashScopeChatFormatter]] = None,
        memory: Optional[InMemoryMemory] = None,
        *,
        default_model_name: str = MODEL_NAME,
        default_api_key: Optional[str] = None,
        default_stream: bool = DEFAULT_STREAM,
        backend: str = DEFAULT_BACKEND,
        max_memory: int = DEFAULT_MAX_MEMORY,
    ) -> None:
        super().__init__()

        self.name = name
        self.sys_prompt = sys_prompt
        self.max_memory = max_memory

        self.safe_name = re.sub(r"[\s<|\\/>]", "_", name)

        if model is not None:
            self.model = model

            if formatter is not None:
                self.formatter = formatter
            else:
                if isinstance(model, OpenAIChatModel):
                    self.formatter = OpenAIChatFormatter()
                elif isinstance(model, DashScopeChatModel):
                    self.formatter = DashScopeChatFormatter()
                else:
                    raise ValueError(
                        f"Unknown model type {type(model)}; "
                        f"please provide a formatter explicitly."
                    )
        else:
            backend = backend.lower()

            if backend == "openai":
                self.model = OpenAIChatModel(
                    model_name=default_model_name,
                    api_key=default_api_key or OPENAI_API_KEY,
                    stream=default_stream,
                )
                self.formatter = formatter or OpenAIChatFormatter()

            elif backend == "dashscope":
                self.model = DashScopeChatModel(
                    model_name=default_model_name,
                    api_key=default_api_key or DASH_KEY,
                    stream=default_stream,
                )
                self.formatter = formatter or DashScopeChatFormatter()

            else:
                raise ValueError(f"Unsupported backend: {backend}")

        self.memory = memory or InMemoryMemory()

    async def _memorize(self, maybe_msgs: Union[Msg, List[Msg], None]) -> None:
        if maybe_msgs is None:
            return

        if isinstance(maybe_msgs, list):
            await self.memory.add(maybe_msgs)
        else:
            await self.memory.add(maybe_msgs)

        if self.max_memory is not None and self.max_memory > 0:
            if hasattr(self.memory, "size") and hasattr(self.memory, "delete"):
                size = await self.memory.size()
                extra = size - self.max_memory
                if extra > 0:
                    await self.memory.delete(list(range(extra)))

    async def reply(self, msg: Union[Msg, List[Msg], None]) -> Msg:
        if isinstance(msg, list):
            user_msgs = msg
        else:
            user_msgs = [msg]

        prompt = await self.formatter.format(
            [
                Msg(name="system", content=self.sys_prompt, role="system"),
                *user_msgs,
            ]
        )

        response = await self.model(prompt)

        out = Msg(
            name=self.safe_name,
            content=response.content,
            role="assistant",
        )

        await self.print(out)
        return out

    async def observe(self, msg: Union[Msg, List[Msg], None]) -> None:
        await self._memorize(msg)

    async def handle_interrupt(self) -> Msg:
        out = Msg(
            name=self.safe_name,
            role="assistant",
            content="I noticed you interrupted my reply. How can I help?",
        )
        await self.print(out)
        return out


class LLMAgent(MyAgent):
    def __init__(
        self,
        name: str = "Friday",
        sys_prompt: str = "You are an AI assistant named Friday.",
        model: Optional[Union[OpenAIChatModel, DashScopeChatModel]] = None,
        formatter: Optional[Union[OpenAIChatFormatter, DashScopeChatFormatter]] = None,
        memory: Optional[InMemoryMemory] = None,
        *,
        default_model_name: str = MODEL_NAME,
        default_api_key: Optional[str] = None,
        default_stream: bool = DEFAULT_STREAM,
        backend: str = DEFAULT_BACKEND,
        max_memory: int = DEFAULT_MAX_MEMORY,
    ) -> None:
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model=model,
            formatter=formatter,
            memory=memory,
            default_model_name=default_model_name,
            default_api_key=default_api_key,
            default_stream=default_stream,
            backend=backend,
            max_memory=max_memory,
        )

    async def reply(self, msg: Union[Msg, List[Msg], None]) -> Msg:
        await self._memorize(msg)

        history = await self.memory.get_memory()

        if isinstance(msg, list):
            user_msgs = msg
        else:
            user_msgs = [msg]

        prompt_msgs = [
            Msg(name="system", content=self.sys_prompt, role="system"),
            *history,
            *user_msgs,
        ]

        prompt = await self.formatter.format(prompt_msgs)
        response = await self.model(prompt)

        out = Msg(
            name=self.safe_name,
            content=response.content,
            role="assistant",
        )

        await self._memorize(out)

        await self.print(out)
        return out
