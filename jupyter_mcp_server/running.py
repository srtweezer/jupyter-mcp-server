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

#: A request the kernel has shown no sign of receiving for this long, while it
#: sits idle and answers ``kernel_info``, is held in the execution stack.
NOT_STARTED_AFTER_S = 10.0

#: How long the kernel gets to answer ``kernel_info`` in that check.
PROBE_TIMEOUT_S = 5.0

#: After a reset, how long a request that slipped out just before it has to
#: show up as kernel activity.
SETTLE_S = 0.5

#: ``ename`` of the result a reset gives each request it drops: one the kernel
#: never received, and one it may have (a restart, a reset for any other reason).
NOT_STARTED = "NotStarted"
DROPPED = "ExecutionDropped"


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
               kernel_state: Optional[Callable[[str], str]] = None,
               kernel_manager: Any = None) -> None:
        """Record ``execution`` and collect its result when it arrives.

        ``on_done`` receives the execution and the stack's raw result, for a
        caller that must still do something with the outputs (file mode writes
        them back into the notebook). ``kernel_state`` maps a kernel id to its
        execution state; with it, a kernel that sits idle with no result
        arriving ends the follow (see ``IDLE_WITHOUT_RESULT_S``), and with
        ``kernel_manager`` too the follow says whether the kernel ever
        received the request (see ``drop_if_never_started``).
        """
        kernel = execution.kernel_id
        self._running.setdefault(kernel, {})[execution.request_id] = execution
        task = asyncio.ensure_future(self._collect(
            execution_stack, execution, on_done, kernel_state, kernel_manager))
        self._tasks[execution.request_id] = task
        task.add_done_callback(
            lambda _t, rid=execution.request_id: self._tasks.pop(rid, None))

    async def _collect(self, execution_stack: Any, execution: Execution,
                       on_done, kernel_state=None, kernel_manager=None) -> None:
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
                            # The stack's worker is still waiting, and would
                            # hold every later request: reset it, which also
                            # answers every request it held, this one included.
                            reason = None
                            if state != "missing" and kernel_manager is not None:
                                reason = await drop_if_never_started(
                                    execution_stack, kernel_manager,
                                    execution.kernel_id, execution.started_at)
                            if reason is None:
                                reason = (f"the kernel is {state} and no result "
                                          f"arrived: it was restarted or shut "
                                          f"down, does not answer, or its reply "
                                          f"never reached the server's execution "
                                          f"queue")
                                await reset_stack(execution_stack,
                                                  execution.kernel_id, reason=reason)
                            result = _take(execution_stack, execution)
                            if result is None:
                                self._finish(execution, "lost", error=reason)
                                return
                            break
                await asyncio.sleep(FOLLOW_INTERVAL_S)
            else:
                self._finish(execution, "lost",
                             error="no longer followed (running too long)")
                return
            self._finish_with(execution, result)
            if on_done is not None and execution.status != "lost":
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
            if not isinstance(info, dict):          # nbmodel's own: a bare string
                info = {"evalue": str(info)}
            if info.get("ename") in (NOT_STARTED, DROPPED):
                # Dropped by a reset: the stack no longer has it.
                self._finish(execution, "lost", error=info.get("evalue", ""))
                return
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

#: The first characters of a result that says the kernel never received the
#: request, so nothing ran. A client may test for it; keep it stable.
NOT_STARTED_MARKER = "[NOT STARTED]"


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


def not_started_message(cell_index: Optional[int], reason: str) -> str:
    """The tool result for an execution the kernel never received."""
    where = f"Cell {cell_index}" if cell_index is not None else "The code"
    return (f"{NOT_STARTED_MARKER} {where} did not run: {reason} Nothing ran, "
            f"so it is safe to run it again.")


async def reset_stack(execution_stack: Any, kernel_id: str,
                      timeout: float = 5.0, reason: str = "") -> None:
    """Drop the ExecutionStack's worker and client for a restarted kernel.

    The worker waits for the ``idle`` status of the request it is executing,
    without a timeout; a restart kills the code that would have sent it, so
    the worker waits forever and every later request queues behind it.
    ``cancel`` discards the worker, its queue and its client, so the next
    request starts a fresh one (made ready again: ``ready`` is cleared).

    Every request the stack still held is answered with a ``DROPPED`` error
    naming ``reason``. ``cancel`` keeps their entries, which then read as
    pending forever: on 2026-10-06 a cell queued behind a stuck one waited out
    its whole 280 s for a request the reset had already discarded.
    """
    registry().ready.discard(kernel_id)
    if execution_stack is None:
        return
    held = _pending(execution_stack, kernel_id)
    try:
        await _cancel(execution_stack, kernel_id, timeout)
    finally:
        _answer(execution_stack, kernel_id, held, DROPPED, reason or (
            "the server's execution queue for this kernel was reset before "
            "the request finished."))


async def drop_if_never_started(execution_stack: Any, kernel_manager: Any,
                                kernel_id: str, since: float,
                                timeout: float = 1.0) -> Optional[str]:
    """Reset the stack if the kernel never received what was submitted at ``since``.

    The case of 2026-10-01 and 2026-10-06: nbmodel's worker held a request
    without ever sending it, the kernel sat idle, and the request "timed out"
    as if the cell were slow; the next one queued behind it. A kernel that is
    idle, has done nothing tracked since ``since`` and answers ``kernel_info``
    now (so it is reading its shell socket) was never sent the request.

    ``timeout`` bounds the reset, whose wait for the held request's queue
    entry always runs out (see ``_cancel``): kept short, since the caller waits.

    Returns the reason every held request was answered with, or None when
    that cannot be shown (nothing is done then). Each request is answered
    ``NOT_STARTED``, unless the kernel became active while the worker was
    being cancelled: the request may have gone out at the last moment, and is
    answered ``DROPPED`` with a reason that says so.
    """
    if execution_stack is None or kernel_id in _DROPPING \
            or not quiet_since(kernel_manager, kernel_id, since):
        return None
    # A request queued behind the stuck one reaches here too; one drop answers
    # both, so the second leaves it to the first.
    _DROPPING.add(kernel_id)
    try:
        return await _drop_if_never_started(execution_stack, kernel_manager,
                                            kernel_id, since, timeout)
    finally:
        _DROPPING.discard(kernel_id)


#: Kernels a ``drop_if_never_started`` is at work on.
_DROPPING: set = set()


async def _drop_if_never_started(execution_stack, kernel_manager, kernel_id,
                                 since, timeout) -> Optional[str]:
    if not await kernel_answers(kernel_manager, kernel_id):
        return None
    if not quiet_since(kernel_manager, kernel_id, since):
        return None
    registry().ready.discard(kernel_id)
    waited = time.time() - since
    held = _pending(execution_stack, kernel_id)
    # Where the worker holds it, read before the cancel unwinds it: the cause
    # is still unknown, and the server's log is only its terminal.
    holder = worker_state(execution_stack, kernel_id)
    logger.warning("Kernel %s: request(s) %s held %.0f s without reaching the "
                   "kernel; %s", kernel_id, held, waited, holder)
    try:
        await _cancel(execution_stack, kernel_id, timeout)
    except asyncio.CancelledError:
        _answer(execution_stack, kernel_id, held, DROPPED,
                "the server's execution queue for this kernel was reset.")
        raise
    await asyncio.sleep(SETTLE_S)
    if quiet_since(kernel_manager, kernel_id, since):
        reason = (f"kernel {kernel_id} received nothing in the {waited:.0f} s "
                  f"after it was submitted, while idle and answering: the "
                  f"server's execution queue held it and never sent it. The "
                  f"queue has been reset. (Server side: {holder}.)")
        _answer(execution_stack, kernel_id, held, NOT_STARTED, reason)
    else:
        reason = (f"the server's execution queue for kernel {kernel_id} held "
                  f"this request for {waited:.0f} s while the kernel sat idle, "
                  f"and was reset; the kernel became active just as it was, so "
                  f"the request may have reached it at the last moment and "
                  f"be running with nobody collecting its output. Check "
                  f"kernel_status before running it again. (Server side: "
                  f"{holder}.)")
        _answer(execution_stack, kernel_id, held, DROPPED, reason)
    return reason


def worker_state(execution_stack: Any, kernel_id: str) -> str:
    """Where nbmodel's worker for ``kernel_id`` is, in one line.

    A held request is either queued behind a worker suspended in some await
    (the chain names it, down to the future it waits on), or queued for a
    worker that has died -- ``put`` never replaces a finished one, so every
    later request on the kernel waits forever.
    """
    workers = getattr(execution_stack, "_ExecutionStack__workers", None)
    worker = workers.get(kernel_id) if isinstance(workers, dict) else None
    if worker is None:
        return "no worker"
    if worker.done():
        if worker.cancelled():
            return "worker finished (cancelled)"
        return f"worker finished, raising {worker.exception()!r}"
    links, coro, uid = [], worker.get_coro(), None
    while coro is not None and len(links) < 20:
        frame = getattr(coro, "cr_frame", None) or getattr(coro, "gi_frame", None)
        if frame is None:
            links.append(type(coro).__name__)
            break
        if uid is None and "uid" in frame.f_locals:
            uid = frame.f_locals["uid"]
        links.append(f"{frame.f_code.co_name}:{frame.f_lineno}")
        nxt = getattr(coro, "cr_await", None) or getattr(coro, "gi_yieldfrom", None)
        if nxt is None:
            break
        if not hasattr(nxt, "cr_frame") and not hasattr(nxt, "gi_frame"):
            links.append(repr(nxt)[:120])
            break
        coro = nxt
    return f"worker on request {uid}, awaiting {' > '.join(links)}"


def quiet_since(kernel_manager: Any, kernel_id: str, since: float) -> bool:
    """The kernel is idle and has done nothing tracked since ``since`` (epoch s).

    jupyter_server moves ``last_activity`` on every IOPub message answering a
    tracked request, an execute among them, so a kernel sent an execute after
    ``since`` cannot pass. ``kernel_info`` is untracked: asking the kernel
    whether it answers does not move it.
    """
    if kernel_manager is None:
        return False
    try:
        kernel = kernel_manager.get_kernel(kernel_id)
    except KeyError:
        return False
    if getattr(kernel, "execution_state", None) != "idle":
        return False
    last = getattr(kernel, "last_activity", None)
    return last is not None and last.timestamp() < since


async def kernel_answers(kernel_manager: Any, kernel_id: str,
                         timeout: Optional[float] = None) -> bool:
    """Whether the kernel answers ``kernel_info`` within ``timeout``, on a
    client of its own: alive, and reading what is sent to it."""
    from inspect import isawaitable
    from jupyter_mcp_server.utils import with_own_identity

    timeout = PROBE_TIMEOUT_S if timeout is None else timeout
    try:
        client = with_own_identity(kernel_manager.get_kernel(kernel_id).client())
    except Exception:  # noqa: BLE001 - no kernel, no answer
        return False
    try:
        started = client.start_channels()
        if isawaitable(started):
            await started
        ready = client.wait_for_ready(timeout=timeout)
        if isawaitable(ready):
            await ready
        return True
    except Exception:  # noqa: BLE001 - no answer is the answer
        return False
    finally:
        try:
            client.stop_channels()
        except Exception:  # noqa: BLE001
            pass


def _results_of(execution_stack: Any, kernel_id: str) -> Dict[str, Any]:
    """nbmodel's result table for ``kernel_id`` (request id -> result).

    Private to the stack, and reached as ``utils.execution_stack_of`` reaches
    the stack itself; an nbmodel without it gives an empty table.
    """
    table = getattr(execution_stack, "_ExecutionStack__execution_results", None)
    return table.get(kernel_id, {}) if isinstance(table, dict) else {}


def _pending(execution_stack: Any, kernel_id: str) -> List[str]:
    """The requests on ``kernel_id`` the stack has no result for yet."""
    try:
        from jupyter_server_nbmodel.execution_stack import NO_RESULT
    except ImportError:
        return []
    return [uid for uid, result in _results_of(execution_stack, kernel_id).items()
            if result is NO_RESULT]


def _answer(execution_stack: Any, kernel_id: str, uids: List[str],
            ename: str, reason: str) -> None:
    """Give each of ``uids`` that is still unanswered an error result.

    The cancelled worker's own answer to the request it held, a bare string
    (``{"error": "CancelledError()"}``), counts as unanswered: read as an
    error it says nothing.
    """
    try:
        from jupyter_server_nbmodel.execution_stack import NO_RESULT
    except ImportError:
        return
    results = _results_of(execution_stack, kernel_id)
    for uid in uids:
        if uid not in results:
            continue
        current = results[uid]
        if current is NO_RESULT or (isinstance(current, dict)
                                    and isinstance(current.get("error"), str)):
            results[uid] = {"error": {"ename": ename, "evalue": reason}}


def _take(execution_stack: Any, execution: Execution) -> Any:
    """The stack's result for ``execution`` (read once), or None."""
    try:
        return execution_stack.get(execution.kernel_id, execution.request_id)
    except ValueError:
        return None


async def _cancel(execution_stack: Any, kernel_id: str, timeout: float) -> None:
    """``ExecutionStack.cancel``, with what it raises on success absorbed."""
    try:
        await execution_stack.cancel(kernel_id, timeout=timeout)
    except asyncio.CancelledError:
        # cancel() awaits the worker it has just cancelled, and awaiting a
        # cancelled task raises CancelledError here too. That one is expected;
        # a cancellation of *this* task is not, and must go on.
        current = asyncio.current_task()
        if current is not None and current.cancelling():
            raise
    except ValueError as error:
        # The cancelled worker drains its queue with task_done() alone, which
        # never empties it, so with a request queued behind the stuck one the
        # drain overshoots and raises. cancel() has still dropped the worker,
        # queue and client in its finally blocks: the reset worked.
        if "task_done" not in str(error):
            logger.warning("Resetting the execution stack of kernel %s: %s",
                           kernel_id, error)
        else:
            logger.debug("Execution stack of kernel %s reset (%s)",
                         kernel_id, error)
    except TimeoutError:
        # With nothing queued behind the held request the drain does nothing,
        # and cancel()'s queue.join() waits for a task_done() that never
        # comes. Its finally blocks have still dropped the queue and client.
        logger.debug("Execution stack of kernel %s reset (queue join timed out)",
                     kernel_id)
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
