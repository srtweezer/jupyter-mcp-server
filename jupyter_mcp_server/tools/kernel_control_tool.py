# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""What a notebook's kernel is doing, and stopping it.

``execute_cell`` waits a bounded time and leaves a slow cell running (see
``jupyter_mcp_server.running``). These two tools are the rest of that story:
``kernel_status`` says whether the kernel is still busy, with what, and what a
followed execution printed once it finished; ``interrupt_kernel`` stops the
running cell and keeps the kernel's state, which ``restart_notebook`` does not.

Both answer in JSON, so a client can read them as well as a model.
"""

import asyncio
import inspect
import json
import logging
import time
from typing import Any, Dict, Optional

from jupyter_mcp_server.running import kernel_state_of, registry
from jupyter_mcp_server.tools._base import ServerMode

logger = logging.getLogger(__name__)

#: How long interrupt_kernel waits to see the kernel settle before answering.
INTERRUPT_SETTLE_S = 5.0

#: Finished executions listed by kernel_status, newest first.
FINISHED_LISTED = 3


def _resolve(notebook_manager, notebook_name: str):
    """(name, kernel_id), or (None, reason)."""
    name = notebook_name or notebook_manager.get_current_notebook()
    if not name:
        return None, ("No notebook is active. Call use_notebook first, or pass "
                      "notebook_name.")
    if name not in notebook_manager:
        return None, (f"Notebook '{name}' is not connected. Connected notebooks: "
                      f"{list(notebook_manager.list_all_notebooks().keys())}")
    kernel_id = notebook_manager.get_kernel_id(name)
    if not kernel_id:
        return None, f"Notebook '{name}' has no kernel."
    return name, kernel_id


def _kernel_state(kernel_manager, kernel_id: str) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "execution_state": kernel_state_of(kernel_manager, kernel_id)}
    if state["execution_state"] == "missing":
        return state
    kernel = kernel_manager.get_kernel(kernel_id)
    last = getattr(kernel, "last_activity", None)
    if last is not None:
        try:
            state["last_activity"] = last.isoformat()
        except AttributeError:
            state["last_activity"] = str(last)
    return state


def kernel_status(mode, notebook_manager, kernel_manager,
                  notebook_name: str = "") -> str:
    if mode != ServerMode.JUPYTER_SERVER or kernel_manager is None:
        return json.dumps({"error": "kernel_status needs the Jupyter server "
                                    "extension deployment."})
    name, kernel_id = _resolve(notebook_manager, notebook_name)
    if name is None:
        return json.dumps({"error": kernel_id})
    now = time.time()
    tracked = registry()
    answer: Dict[str, Any] = {"notebook": name, "kernel_id": kernel_id}
    answer.update(_kernel_state(kernel_manager, kernel_id))
    answer["running"] = [e.describe(now) for e in tracked.running(kernel_id)]
    # Executions a tool call is waiting on right now: busy, but not news.
    answer["waited_on"] = len(tracked.waited(kernel_id))
    answer["finished"] = [e.describe(now) for e in
                          reversed(tracked.finished(kernel_id)[-FINISHED_LISTED:])]
    if (answer["execution_state"] == "busy" and not answer["running"]
            and not answer["waited_on"]):
        answer["note"] = ("The kernel is busy with something no tool call is "
                          "waiting on — code whose wait timed out in "
                          "execute_code, or a cell run from JupyterLab.")
    return json.dumps(answer)


async def interrupt_kernel(mode, notebook_manager, kernel_manager,
                           notebook_name: str = "") -> str:
    if mode != ServerMode.JUPYTER_SERVER or kernel_manager is None:
        return json.dumps({"error": "interrupt_kernel needs the Jupyter server "
                                    "extension deployment."})
    name, kernel_id = _resolve(notebook_manager, notebook_name)
    if name is None:
        return json.dumps({"error": kernel_id})

    tracked = registry()
    before = {e.request_id for e in tracked.running(kernel_id)}
    state_before = _kernel_state(kernel_manager, kernel_id)["execution_state"]
    try:
        # Synchronous on some kernel managers, a coroutine on others.
        outcome = kernel_manager.interrupt_kernel(kernel_id)
        if inspect.isawaitable(outcome):
            await outcome
    except Exception as error:  # noqa: BLE001
        logger.error("Interrupting kernel %s failed: %s", kernel_id, error)
        return json.dumps({"notebook": name, "kernel_id": kernel_id,
                           "error": f"interrupt failed: {error}"})

    # Give the kernel a moment to raise KeyboardInterrupt and the follower a
    # moment to collect it, so the answer says what actually happened.
    deadline = time.time() + INTERRUPT_SETTLE_S
    while time.time() < deadline:
        still = {e.request_id for e in tracked.running(kernel_id)}
        state = _kernel_state(kernel_manager, kernel_id)["execution_state"]
        if not (still & before) and state != "busy":
            break
        await asyncio.sleep(0.2)

    now = time.time()
    stopped = [e.describe(now) for e in tracked.finished(kernel_id)
               if e.request_id in before]
    answer: Dict[str, Any] = {
        "notebook": name, "kernel_id": kernel_id,
        "interrupted": True, "state_before": state_before,
        "stopped": stopped,
        "still_running": [e.describe(now) for e in tracked.running(kernel_id)],
    }
    answer.update(_kernel_state(kernel_manager, kernel_id))
    if answer["execution_state"] == "busy":
        answer["note"] = (f"Still busy {INTERRUPT_SETTLE_S:g} s after the "
                          f"interrupt: the code may be in a call that ignores "
                          f"KeyboardInterrupt. restart_notebook stops it for "
                          f"certain, and loses the kernel's variables.")
    else:
        answer["note"] = "Variables and imports are kept; only the running code stopped."
    return json.dumps(answer)
