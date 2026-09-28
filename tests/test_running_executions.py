# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""A cell that outlives the wait for it keeps running, and is accounted for.

The timeout path used to call ``ExecutionStack.cancel`` without awaiting it, so
the cell ran on while nothing knew: the result read as a plain error, the next
execute queued silently behind it and timed out in its turn, and there was no
way to ask what the kernel was doing or to stop it short of a restart.

Section A drives ``RunningExecutions`` with a fake stack. Section B runs the
real thing against JupyterLab with the extension (JUPYTER_SERVER mode only:
the MCP_SERVER mode keeps its own timeout path).
"""

import asyncio
import json
import os

import pytest

from jupyter_mcp_server import running
from jupyter_mcp_server.running import (
    Execution, KERNEL_BUSY_MARKER, RunningExecutions, STILL_RUNNING_MARKER,
    busy_executions)


###############################################################################
# Section A — the registry, with a fake stack
###############################################################################

class FakeStack:
    """``ExecutionStack.get``: None while pending, the result exactly once."""

    def __init__(self):
        self.results = {}
        self.forgotten = set()

    def get(self, kernel_id, uid):
        if uid in self.forgotten:
            raise ValueError(f"Execution request {uid} does not exists.")
        return self.results.pop(uid, None)


class FakeKernelManager:
    def __init__(self, state="busy"):
        self.state = state

    def get_kernel(self, kernel_id):
        if self.state == "missing":
            raise KeyError(kernel_id)
        return type("K", (), {"execution_state": self.state})()


@pytest.fixture(autouse=True)
def fast_follow(monkeypatch):
    monkeypatch.setattr(running, "FOLLOW_INTERVAL_S", 0.01)
    monkeypatch.setattr(running, "IDLE_WITHOUT_RESULT_S", 0.05)


async def _until(predicate, limit=2.0):
    for _ in range(int(limit / 0.01)):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never became true")


@pytest.mark.asyncio
async def test_a_followed_execution_finishes_with_its_outputs():
    registry, stack = RunningExecutions(), FakeStack()
    registry.follow(stack, Execution("k", "r1", notebook="nb", cell_index=3,
                                     code="load()\nmore()"))
    assert [e.request_id for e in registry.running("k")] == ["r1"]

    stack.results["r1"] = {"outputs": json.dumps([
        {"output_type": "stream", "name": "stdout", "text": "loaded 12 GB\n"}])}
    await _until(lambda: registry.finished("k"))

    done = registry.finished("k")[0].describe()
    assert registry.running("k") == []
    assert done["status"] == "ok" and done["cell_index"] == 3
    assert "loaded 12 GB" in done["outputs"][0]
    assert done["code_first_line"] == "load()"


@pytest.mark.asyncio
async def test_an_error_is_reported_as_one():
    registry, stack = RunningExecutions(), FakeStack()
    registry.follow(stack, Execution("k", "r1"))
    stack.results["r1"] = {"error": {"ename": "KeyboardInterrupt", "evalue": ""}}
    await _until(lambda: registry.finished("k"))
    assert registry.finished("k")[0].error.startswith("KeyboardInterrupt")


@pytest.mark.asyncio
async def test_an_exception_in_the_code_is_an_error_too():
    """The code's own exception (an interrupt's KeyboardInterrupt) arrives as
    an error *output*; reading only the stack's error field called it ok."""
    registry, stack = RunningExecutions(), FakeStack()
    registry.follow(stack, Execution("k", "r1"))
    stack.results["r1"] = {"outputs": [
        {"output_type": "error", "ename": "KeyboardInterrupt", "evalue": "",
         "traceback": ["KeyboardInterrupt"]}]}
    await _until(lambda: registry.finished("k"))
    done = registry.finished("k")[0]
    assert done.status == "error" and done.error.startswith("KeyboardInterrupt")


@pytest.mark.asyncio
async def test_a_request_the_stack_forgot_is_lost_not_running_forever():
    registry, stack = RunningExecutions(), FakeStack()
    stack.forgotten.add("r1")
    registry.follow(stack, Execution("k", "r1"))
    await _until(lambda: registry.finished("k"))
    assert registry.finished("k")[0].status == "lost"


@pytest.mark.asyncio
async def test_an_idle_kernel_with_no_result_ends_the_follow():
    """Restarted from JupyterLab's menu: the result will never arrive."""
    registry, stack, km = RunningExecutions(), FakeStack(), FakeKernelManager("idle")
    registry.follow(stack, Execution("k", "r1"),
                    kernel_state=lambda kid: km.get_kernel(kid).execution_state)
    await _until(lambda: registry.finished("k"))
    assert registry.finished("k")[0].status == "lost"


@pytest.mark.asyncio
async def test_a_busy_kernel_is_followed_however_long_it_takes():
    registry, stack, km = RunningExecutions(), FakeStack(), FakeKernelManager("busy")
    registry.follow(stack, Execution("k", "r1"),
                    kernel_state=lambda kid: km.get_kernel(kid).execution_state)
    await asyncio.sleep(0.2)            # four times IDLE_WITHOUT_RESULT_S
    assert registry.running("k"), "a busy kernel must not be given up on"


@pytest.mark.asyncio
async def test_forgetting_a_kernel_marks_its_executions_lost():
    registry, stack = RunningExecutions(), FakeStack()
    registry.follow(stack, Execution("k", "r1"))
    registry.forget_kernel("k", "restarted")
    assert registry.running("k") == []
    assert registry.finished("k")[0].error == "restarted"


@pytest.mark.asyncio
async def test_a_record_the_kernel_contradicts_does_not_refuse_work(monkeypatch):
    """Refusing on a stale record would lock the notebook until the follower
    noticed; only a kernel that says it is busy is busy."""
    registry, stack = RunningExecutions(), FakeStack()
    monkeypatch.setattr(running, "_REGISTRY", registry)
    registry.follow(stack, Execution("k", "r1"))

    assert busy_executions(FakeKernelManager("busy"), "k")
    assert busy_executions(FakeKernelManager("idle"), "k") == []


###############################################################################
# Section B — against JupyterLab with the extension
###############################################################################

from .test_common import MCPClient  # noqa: E402

NOTEBOOK = "running_executions_test.ipynb"


@pytest.fixture
def lab_client(jupyter_server_with_extension):
    from .conftest import JUPYTER_TOKEN
    yield MCPClient(jupyter_server_with_extension, token=JUPYTER_TOKEN)
    path = os.path.join("dev", "content", NOTEBOOK)
    if os.path.exists(path):
        os.remove(path)


def _text(structured):
    result = (structured or {}).get("result")
    if isinstance(result, list):
        return "\n".join(str(r) for r in result)
    return str(result)


async def _status(client):
    result = await client._session.call_tool("kernel_status", arguments={})
    return json.loads(client._extract_text_content(result))


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_slow_cell_keeps_running_and_reports_back(lab_client):
    async with lab_client:
        await lab_client.use_notebook("slow", NOTEBOOK, mode="create")
        await lab_client.insert_cell(0, "code",
                                     "import time\nprint('started', flush=True)\n"
                                     "time.sleep(6)\nprint('done')")
        first = _text(await lab_client.execute_cell(0, timeout_seconds=2))
        assert first.startswith(STILL_RUNNING_MARKER), first
        assert "NOT stopped" in first

        status = await _status(lab_client)
        assert status["execution_state"] == "busy"
        assert status["running"][0]["cell_index"] == 0

        refused = await lab_client._session.call_tool(
            "execute_code", arguments={"code": "1 + 1", "timeout": 5})
        assert KERNEL_BUSY_MARKER in lab_client._extract_text_content(refused)

        for _ in range(60):
            status = await _status(lab_client)
            if not status["running"]:
                break
            await asyncio.sleep(0.5)
        finished = status["finished"][0]
        assert finished["status"] == "ok"
        assert "done" in finished["outputs"][0]
        assert finished["elapsed_s"] >= 5, "timed from the submit, not the timeout"


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_interrupt_stops_the_cell_and_keeps_the_variables(lab_client):
    async with lab_client:
        await lab_client.use_notebook("interrupt", NOTEBOOK, mode="create")
        await lab_client.insert_cell(0, "code", "kept = 41")
        await lab_client.insert_cell(1, "code", "import time\ntime.sleep(60)")
        await lab_client.execute_cell(0, timeout_seconds=10)
        first = _text(await lab_client.execute_cell(1, timeout_seconds=2))
        assert first.startswith(STILL_RUNNING_MARKER), first

        result = await lab_client._session.call_tool("interrupt_kernel", arguments={})
        answer = json.loads(lab_client._extract_text_content(result))
        assert answer["execution_state"] != "busy", answer
        assert answer["stopped"][0]["error"].startswith("KeyboardInterrupt"), answer

        after = await lab_client.execute_code("kept + 1", timeout=10)
        assert "42" in _text(after)
