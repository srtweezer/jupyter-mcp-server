# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""What the cell tools do when no notebook is open.

They used to act on whatever ``get_current_notebook_context`` returned, and that
included the configured document — including its placeholder default. Two
placeholders fed it: the extension's ``document_id`` trait, which shipped as
``"notebook.ipynb"`` and was auto-enrolled as the *active* notebook at startup,
and ``execute_code``, which invented the same string when it started a kernel on
demand. So a server that had opened nothing looked exactly like one that had:

    insert_cell -> [Errno 2] No such file or directory: <root>/notebook.ipynb

an errno about a file the caller had never mentioned, hours after startup,
naming nothing to do about it. ``list_notebooks`` showed the phantom with a tick
beside it, so even looking did not help.

These run without a server: the defect was in what the manager and the config
agree the active notebook is, which is decided before any I/O happens.
"""

import pytest

from jupyter_mcp_server.config import get_config
from jupyter_mcp_server.notebook_manager import NotebookManager
from jupyter_mcp_server.utils import (
    get_current_notebook_context,
    require_notebook_path,
)


@pytest.fixture(autouse=True)
def no_configured_document(monkeypatch):
    """A server nobody configured a document for — the CsGUI deployment, and
    the default everywhere since ``JupyterMCPConfig.document_id`` is None."""
    config = get_config()
    monkeypatch.setattr(config, "document_id", None, raising=False)
    monkeypatch.setattr(config, "runtime_id", None, raising=False)


def test_an_empty_manager_yields_no_notebook():
    assert get_current_notebook_context(NotebookManager()) == (None, None)


def test_asking_for_the_path_says_what_to_do():
    with pytest.raises(ValueError) as raised:
        require_notebook_path(NotebookManager())

    message = str(raised.value)
    assert "use_notebook" in message, "does not name the call that fixes it"
    assert "list_notebooks" in message, "does not say how to see what is open"


def test_a_kernel_without_a_notebook_is_not_a_notebook():
    """``execute_code`` starts a kernel on demand and registers it so the next
    call can find it. That entry holds a kernel and nothing else — claiming it
    has a notebook is what made the following ``insert_cell`` open a file."""
    manager = NotebookManager()
    manager.add_notebook("default", {"id": "kernel-1"},
                         server_url="local", token=None, path=None)

    _, kernel_id = get_current_notebook_context(manager)
    assert kernel_id == "kernel-1", "the kernel must still be found"

    with pytest.raises(ValueError):
        require_notebook_path(manager)


def test_execute_code_invents_no_notebook_path():
    """The literal is gone, not merely unused: it was passed as ``path=`` and
    became the active notebook."""
    from pathlib import Path

    import jupyter_mcp_server.tools.execute_code_tool as module

    assert "notebook.ipynb" not in Path(module.__file__).read_text()


def test_the_extension_has_no_default_document():
    """Whatever this holds is auto-enrolled as ``default`` and made active."""
    from jupyter_mcp_server.jupyter_extension.extension import (
        JupyterMCPServerExtensionApp,
    )

    assert JupyterMCPServerExtensionApp.document_id.default_value == ""


def test_an_open_notebook_is_still_found():
    """The refusal must not have cost the ordinary case."""
    manager = NotebookManager()
    manager.add_notebook("analysis", {"id": "kernel-2"},
                         server_url="local", token=None, path="runs/analysis.ipynb")

    assert require_notebook_path(manager) == "runs/analysis.ipynb"
    assert require_notebook_path(manager, "analysis") == "runs/analysis.ipynb"


CELL_TOOLS = [
    "insert_cell_tool", "read_cell_tool", "delete_cell_tool", "move_cell_tool",
    "edit_cell_source_tool", "overwrite_cell_source_tool", "execute_cell_tool",
]


@pytest.mark.parametrize("module_name", CELL_TOOLS)
def test_every_cell_tool_asks_for_a_notebook_rather_than_a_context(module_name):
    """The refusal is only worth anything where the path is used.

    Each of these took ``get_current_notebook_context(...)[0]`` and handed it
    straight to ``open()``, so any one of them left behind keeps the old
    failure — and keeps it invisible, since six tools out of seven behaving is
    indistinguishable from all of them behaving until somebody uses the seventh.
    """
    from importlib import import_module
    from pathlib import Path

    source = Path(import_module(
        f"jupyter_mcp_server.tools.{module_name}").__file__).read_text()

    assert "require_notebook_path(" in source


def test_insert_cell_refuses_rather_than_opening_something():
    """Through the tool, not the helper: this is the call the user made."""
    import asyncio

    from jupyter_mcp_server.tools._base import ServerMode
    from jupyter_mcp_server.tools.insert_cell_tool import InsertCellTool

    with pytest.raises(ValueError) as raised:
        asyncio.run(InsertCellTool().execute(
            mode=ServerMode.JUPYTER_SERVER,
            contents_manager=object(),
            notebook_manager=NotebookManager(),
            cell_index=-1, cell_type="code", cell_source="1 + 1",
        ))

    assert "use_notebook" in str(raised.value)


def test_an_unregistered_name_still_names_itself():
    manager = NotebookManager()

    with pytest.raises(ValueError) as raised:
        require_notebook_path(manager, "missing")

    assert "missing" in str(raised.value)
