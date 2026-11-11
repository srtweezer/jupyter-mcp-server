# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""A request that names its notebook acts on that notebook, whatever else is
in flight.

CsGUI runs several agents against one server, each in a notebook (and kernel)
of its own, and each names its notebook on every call. The server used to honour
the name by switching its one *active* notebook for the length of the call and
switching it back -- and the tools await in between, so a second agent's call
landing meanwhile read the first one's notebook: a cell written, or code run,
in another agent's kernel. The name is now held per request
(``NotebookManager.targeting``), and the active notebook is never touched.

Runs without a server: what each request sees is decided in the manager.
"""

import asyncio

import pytest

from jupyter_mcp_server.notebook_manager import NotebookManager
from jupyter_mcp_server.tools._base import ServerMode
from jupyter_mcp_server.tools.unuse_notebook_tool import (
    UnuseNotebookTool, shut_down_kernel,
)


def _manager(*names):
    manager = NotebookManager()
    for name in names:
        manager.add_notebook(name, {"id": f"k-{name}"}, server_url="local",
                             path=f"{name}.ipynb")
    return manager


def test_two_requests_in_flight_each_act_on_the_notebook_they_named():
    manager = _manager("main", "e1", "e2")
    manager.set_current_notebook("main")
    seen = {}

    async def request(name):
        with manager.targeting(name):
            # A tool awaits between reading the notebook and acting on it.
            await asyncio.sleep(0.01)
            first = manager.get_current_notebook()
            await asyncio.sleep(0.01)
            seen[name] = (first, manager.get_current_notebook(),
                          manager.get_kernel_id(manager.get_current_notebook()))

    async def both():
        await asyncio.gather(request("e1"), request("e2"), request(""))

    asyncio.run(both())
    assert seen["e1"] == ("e1", "e1", "k-e1")
    assert seen["e2"] == ("e2", "e2", "k-e2")
    # A request naming nothing gets the active notebook, as before.
    assert seen[""] == ("main", "main", "k-main")
    assert manager.get_current_notebook() == "main"


def test_naming_an_unknown_notebook_says_how_to_open_it():
    manager = _manager("main")
    with pytest.raises(ValueError, match="use_notebook"):
        with manager.targeting("nope"):
            pass


class _Sessions:
    def __init__(self, sessions):
        self.sessions = sessions
        self.deleted = []

    async def list_sessions(self):
        return list(self.sessions)

    async def delete_session(self, session_id):
        self.deleted.append(session_id)


class _Kernels:
    def __init__(self, ids, fail=False):
        self.ids = set(ids)
        self.shut = []
        self.fail = fail

    def __contains__(self, kernel_id):
        return kernel_id in self.ids

    async def shutdown_kernel(self, kernel_id):
        if self.fail:
            raise RuntimeError("busy")
        self.shut.append(kernel_id)
        self.ids.discard(kernel_id)


def test_a_kernel_is_shut_down_with_its_sessions_and_nobody_elses():
    sessions = _Sessions([{"id": "s1", "kernel": {"id": "k-e1"}},
                          {"id": "s2", "kernel": {"id": "k-main"}}])
    kernels = _Kernels({"k-e1", "k-main"})
    assert asyncio.run(shut_down_kernel("k-e1", kernels, sessions)) is True
    assert sessions.deleted == ["s1"]
    assert kernels.shut == ["k-e1"]


def test_a_kernel_that_cannot_be_shut_down_says_why():
    said = asyncio.run(shut_down_kernel("k-e1", _Kernels({"k-e1"}, fail=True)))
    assert said == "RuntimeError: busy"


@pytest.mark.parametrize("shutdown", [True, False])
def test_unuse_shuts_the_kernel_down_only_when_asked(shutdown):
    manager = _manager("main", "e1")
    kernels = _Kernels({"k-e1", "k-main"})
    message = asyncio.run(UnuseNotebookTool().execute(
        mode=ServerMode.JUPYTER_SERVER, notebook_manager=manager,
        kernel_manager=kernels, session_manager=_Sessions([]),
        notebook_name="e1", shutdown_kernel=shutdown))
    assert "e1" not in manager
    assert kernels.shut == (["k-e1"] if shutdown else [])
    assert ("was shut down" in message) is shutdown
