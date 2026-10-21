"""One short episode inside a Docker sandbox, scored by a file the agent writes.

This task exists so a single machine can walk the same path as RL rollout
(agent, sandbox, gateway trajectories) without pulling a SWE-bench image.
"""

from __future__ import annotations

import logging

from pydantic import Field

from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task

logger = logging.getLogger(__name__)

RESULT_PATH = "/workspace/result.txt"


class MiniDockerTaskConfig(TaskConfig):
    name: str = "mini_docker"
    expected_text: str = Field(
        default="hello-from-docker",
        description="Exact text that must appear in /workspace/result.txt, after stripping whitespace.",
    )


@register_task("mini_docker")
class MiniDockerTask(Task):
    """Run the configured agent in Docker, then score the file it was asked to create."""

    name = "mini_docker"
    config_model = MiniDockerTaskConfig

    async def run(self) -> TaskResult:
        cfg: MiniDockerTaskConfig = self.config  # type: ignore[assignment]
        expected = str(cfg.metadata.get("expected_text") or cfg.expected_text).strip()
        logger.info("starting mini_docker task, expecting %r in %s", expected, RESULT_PATH)

        async with self.build_sandbox() as sandbox:
            agent = self.build_agent()
            agent_result = await agent.run(
                sandbox=sandbox,
                messages=cfg.prompt,
                workdir="/",
            )
            observed = ""
            read_error = ""
            try:
                observed = (await sandbox.read_file(RESULT_PATH)).decode("utf-8", errors="replace").strip()
            except Exception as exc:
                read_error = f"{type(exc).__name__}: {exc}"
                logger.info("mini_docker could not read %s: %s", RESULT_PATH, read_error)

        reward = 1.0 if observed == expected else 0.0
        logger.info("mini_docker done: reward=%s observed=%r expected=%r", reward, observed, expected)
        return TaskResult(
            reward=reward,
            accuracy=reward,
            finished=agent_result.finished,
            extra_info={
                "expected_text": expected,
                "observed_text": observed,
                "result_path": RESULT_PATH,
                "read_error": read_error,
                "agent_error": agent_result.info.get("error"),
                "steps": agent_result.info.get("steps"),
                "num_tool_calls": agent_result.info.get("num_tool_calls"),
            },
        )
