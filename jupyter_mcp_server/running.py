# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""Executions that outlived the wait for them.

``execute_cell`` waits a bounded time for a cell. When the wait runs out the
cell is *not* stopped: stopping it is a decision for the caller (see
``interrupt_kernel``), and a long data load that is thrown away at the timeout
has to be redone from the start. So the request stays with the
``ExecutionStack`` and is recorded here, where ``kernel_status`` can say what
the kernel is doing and, once it is done, what came out.

Before this module the timeout path called ``ExecutionStack.cancel`` without
awaiting it -- a coroutine that never ran -- so the cell kept running anyway,
and nothing anywhere knew about it. The next execute queued silently behind it
and timed out in its turn.

One follower task per recorded execution collects its result, because the
stack hands a result out exactly once and keeps it forever otherwise.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Finished executions remembered per kernel, newest last.
FINISHED_KEPT = 5

#: How often a follower asks the stack for a result, in seconds.
FOLLOW_INTERVAL_S = 0.5

#: A follower gives up after this long. The execution may still finish; it is
#: then simply no longer reported.
FOLLOW_LIMIT_S = 7 * 24 * 3600

#: A follower whose kernel has been idle this long with no result arriving
#: stops waiting: the kernel was restarted from somewhere this server does not
#: see (JupyterLab's own menu), and the result will never come.
IDLE_WITHOUT_RESULT_S = 30.0

#: Text output kept per finished execution. Images are counted, not kept.
OUTPUT_CHARS_KEPT = 20_000


@dataclass
class Execution:
    """One execution the caller stopped waiting for."""

    kernel_id: str
    request_id: str
    notebook: str = ""
    cell_index: Optional[int] = None
    code: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    #: "ok", "error" or "lost" (the stack no longer knows the request: the
    #: kernel was restarted or shut down).
    status: str = "running"
    error: str = ""
    outputs: List[str] = field(default_factory=list)
    images: int = 0

    def describe(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = now or time.time()
        end = self.finished_at or now
        first_line = self.code.strip().splitlines()[0] if self.code.strip() else ""
        out: Dict[str, Any] = {
            "request_id": self.request_id,
            "notebook": self.notebook,
            "cell_index": self.cell_index,
            "code_first_line": first_line[:200],
            "started_at": self.started_at,
            "elapsed_s": round(end - self.started_at, 1),
            "status": self.status,
        }
        if self.finished_at is not None:
            out["finished_at"] = self.finished_at
            out["outputs"] = self.outputs
            out["images"] = self.images
            if self.error:
                out["error"] = self.error
        return out


class RunningExecutions:
    """Per kernel: what is still running, and what finished lately."""

    def __init__(self) -> None:
        self._running: Dict[str, Dict[str, Execution]] = {}
        self._finished: Dict[str, Deque[Execution]] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        # Executions a tool call is still waiting on: kernel -> token -> start.
        self._waited: Dict[str, Dict[object, float]] = {}
        # Kernels whose ExecutionStack client has been made ready (see
        # utils.ensure_stack_client_ready). Forgotten on restart.
        self.ready: set = set()

    @contextmanager
    def waiting(self, kernel_id: str):
        """Mark ``kernel_id`` as busy on behalf of a caller that is waiting.

        So a watcher can tell "busy with the cell somebody is waiting for"
        from "busy with something nobody is" (a cell run in JupyterLab, code
        whose execute_code wait ran out) -- only the second is news when the
        kernel goes idle.
        """
        token = object()
        self._waited.setdefault(kernel_id, {})[token] = time.time()
        try:
            yield
        finally:
            self._waited.get(kernel_id, {}).pop(token, None)

    def waited(self, kernel_id: str) -> List[float]:
        """Start times of the executions a caller is waiting on."""
        return sorted(self._waited.get(kernel_id, {}).values())

    # -- recording ----------------------------------------------------------

    def follow(self, execution_stack: Any, execution: Execution,
               on_done: Optional[Callable[[Execution, Any], Awaitable[None]]] = None,
               kernel_state: Optional[Callable[[str], str]] = None) -> None:
        """Record ``execution`` and collect its result when it arrives.

        ``on_done`` receives the execution and the stack's raw result, for a
        caller that must still do something with the outputs (file mode writes
        them back into the notebook). ``kernel_state`` maps a kernel id to its
        execution state; with it, a kernel that sits idle with no result
        arriving ends the follow (see ``IDLE_WITHOUT_RESULT_S``).
        """
        kernel = execution.kernel_id
        self._running.setdefault(kernel, {})[execution.request_id] = execution
        task = asyncio.ensure_future(
            self._collect(execution_stack, execution, on_done, kernel_state))
        self._tasks[execution.request_id] = task
        task.add_done_callback(
            lambda _t, rid=execution.request_id: self._tasks.pop(rid, None))

    async def _collect(self, execution_stack: Any, execution: Execution,
                       on_done, kernel_state=None) -> None:
        result = None
        deadline = time.time() + FOLLOW_LIMIT_S
        idle_since = None
        try:
            while time.time() < deadline:
                if execution.status != "running":
                    return      # forgotten: the kernel was restarted
                try:
                    result = execution_stack.get(execution.kernel_id,
                                                 execution.request_id)
                except ValueError:
                    # The stack forgot the request: restart or shutdown.
                    self._finish(execution, "lost",
                                 error="the kernel was restarted or shut down "
                                       "before the execution finished")
                    return
                if result is not None:
                    break
                if kernel_state is not None:
                    state = kernel_state(execution.kernel_id)
                    if state == "busy":
                        idle_since = None
                    else:
                        idle_since = idle_since or time.time()
                        if (state == "missing"
                                or time.time() - idle_since > IDLE_WITHOUT_RESULT_S):
                            self._finish(execution, "lost", error=(
                                f"the kernel is {state} but no result arrived; "
                                f"it was probably restarted or shut down"))
                            # The stack's worker is still waiting for that
                            # result, and would hold every later request.
                            await reset_stack(execution_stack, execution.kernel_id)
                            return
                await asyncio.sleep(FOLLOW_INTERVAL_S)
            else:
                self._finish(execution, "lost",
                             error="no longer followed (running too long)")
                return
            self._finish_with(execution, result)
            if on_done is not None:
                try:
                    await on_done(execution, result)
                except Exception:  # noqa: BLE001 - reporting must not die
                    logger.exception("on_done failed for %s", execution.request_id)
        except asyncio.CancelledError:
            if execution.status == "running":
                self._finish(execution, "lost", error="follower cancelled")
            raise
        except Exception as error:  # noqa: BLE001
            logger.exception("Following %s failed", execution.request_id)
            self._finish(execution, "lost", error=str(error))

    def _finish_with(self, execution: Execution, result: Any) -> None:
        from jupyter_mcp_server.utils import safe_extract_outputs

        if "error" in result:
            info = result["error"] or {}
            error = f"{info.get('ename', 'Error')}: {info.get('evalue', '')}"
            self._finish(execution, "error", error=error)
            return
        raw = result.get("outputs", [])
        if isinstance(raw, str):
            import json
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = []
        # An exception raised by the code (KeyboardInterrupt from an interrupt
        # among them) arrives as an "error" *output*, not as the stack's error.
        errors = [o for o in raw or []
                  if isinstance(o, dict) and o.get("output_type") == "error"]
        texts, images = [], 0
        for item in safe_extract_outputs(raw or []):
            if isinstance(item, str):
                texts.append(item)
            else:
                images += 1
        joined = "\n".join(texts)
        if len(joined) > OUTPUT_CHARS_KEPT:
            joined = (joined[:OUTPUT_CHARS_KEPT // 2] + "\n[... output trimmed ...]\n"
                      + joined[-OUTPUT_CHARS_KEPT // 2:])
        execution.outputs = [joined] if joined else []
        execution.images = images
        if errors:
            last = errors[-1]
            self._finish(execution, "error",
                         error=f"{last.get('ename', 'Error')}: {last.get('evalue', '')}")
            return
        self._finish(execution, "ok")

    def _finish(self, execution: Execution, status: str, error: str = "") -> None:
        execution.status = status
        execution.error = error
        execution.finished_at = time.time()
        self._running.get(execution.kernel_id, {}).pop(execution.request_id, None)
        self._finished.setdefault(
            execution.kernel_id, deque(maxlen=FINISHED_KEPT)).append(execution)

    def forget_kernel(self, kernel_id: str, reason: str) -> None:
        """Mark everything followed on ``kernel_id`` lost: it was restarted or
        shut down, and its results will never arrive."""
        self.ready.discard(kernel_id)
        for execution in list(self._running.get(kernel_id, {}).values()):
            self._finish(execution, "lost", error=reason)
            task = self._tasks.pop(execution.request_id, None)
            if task is not None:
                task.cancel()

    # -- reading ------------------------------------------------------------

    def running(self, kernel_id: str) -> List[Execution]:
        return sorted(self._running.get(kernel_id, {}).values(),
                      key=lambda e: e.started_at)

    def finished(self, kernel_id: str) -> List[Execution]:
        return list(self._finished.get(kernel_id, ()))


_REGISTRY = RunningExecutions()


def registry() -> RunningExecutions:
    """The server's one registry. Module-level: executions outlive tool calls."""
    return _REGISTRY


class StillRunning(TimeoutError):
    """The wait ran out; the execution continues and is being followed."""

    def __init__(self, execution: Execution, waited_s: float):
        self.execution = execution
        self.waited_s = waited_s
        super().__init__(
            f"did not finish within {waited_s:g} s and is still running")


#: The first characters of a result that says a cell is still running. A client
#: may test for it; keep it stable.
STILL_RUNNING_MARKER = "[STILL RUNNING]"

#: The first characters of a refusal to queue behind a running execution.
KERNEL_BUSY_MARKER = "[KERNEL BUSY]"


def still_running_message(error: StillRunning, partial: List[Any]) -> List[Any]:
    """The tool result for a cell that outlived its wait."""
    execution = error.execution
    where = (f"Cell {execution.cell_index}" if execution.cell_index is not None
             else "The code")
    head = (
        f"{STILL_RUNNING_MARKER} {where} did not finish within "
        f"{error.waited_s:g} s. It was NOT stopped: it is still running on "
        f"kernel {execution.kernel_id} (request {execution.request_id}). "
        f"Anything else run in this kernel waits behind it. kernel_status says "
        f"when it is done and what it printed; interrupt_kernel stops it."
    )
    if partial:
        return [head + " Output so far:"] + list(partial)
    return [head + " No output so far."]


async def reset_stack(execution_stack: Any, kernel_id: str,
                      timeout: float = 5.0) -> None:
    """Drop the ExecutionStack's worker and client for a restarted kernel.

    The worker waits for the ``idle`` status of the request it is executing,
    without a timeout; a restart kills the code that would have sent it, so
    the worker waits forever and every later request queues behind it.
    ``cancel`` discards the worker, its queue and its client, so the next
    request starts a fresh one (made ready again: ``ready`` is cleared).
    """
    registry().ready.discard(kernel_id)
    if execution_stack is None:
        return
    try:
        await execution_stack.cancel(kernel_id, timeout=timeout)
    except asyncio.CancelledError:
        # cancel() awaits the worker it has just cancelled, and awaiting a
        # cancelled task raises CancelledError here too. That one is expected;
        # a cancellation of *this* task is not, and must go on.
        current = asyncio.current_task()
        if current is not None and current.cancelling():
            raise
    except Exception as error:  # noqa: BLE001 - best effort; logged
        logger.warning("Resetting the execution stack of kernel %s: %s",
                       kernel_id, error)


def busy_executions(kernel_manager: Any, kernel_id: str) -> List[Execution]:
    """The followed executions an execute on ``kernel_id`` would queue behind.

    Only while the kernel says it is busy: a record the kernel contradicts is
    stale (restarted from JupyterLab), and refusing on it would lock the agent
    out of the notebook until the follower noticed.
    """
    running = registry().running(kernel_id)
    if not running or kernel_manager is None:
        return running
    if kernel_state_of(kernel_manager, kernel_id) == "busy":
        return running
    return []


def kernel_state_of(kernel_manager: Any, kernel_id: str) -> str:
    """``execution_state`` of a kernel, "missing" if the manager has none."""
    try:
        kernel = kernel_manager.get_kernel(kernel_id)
    except KeyError:
        return "missing"
    return getattr(kernel, "execution_state", "unknown") or "unknown"


def busy_message(kernel_id: str, running: List[Execution]) -> str:
    """The refusal for an execute that would queue behind a followed one."""
    now = time.time()
    parts = []
    for execution in running:
        where = (f"cell {execution.cell_index}" if execution.cell_index is not None
                 else "code")
        parts.append(f"{where} of {execution.notebook or 'the notebook'}, "
                     f"running for {now - execution.started_at:.0f} s")
    return (
        f"{KERNEL_BUSY_MARKER} Kernel {kernel_id} is still running an earlier "
        f"execution ({'; '.join(parts)}), so this one would only queue behind "
        f"it. Nothing was run. Wait for it (kernel_status), or stop it "
        f"(interrupt_kernel) and run again."
    )
