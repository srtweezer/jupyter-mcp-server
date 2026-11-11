# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""Unuse notebook tool implementation."""

import logging
from typing import Any, Optional
from jupyter_server_client import JupyterServerClient
from jupyter_mcp_server.tools._base import BaseTool, ServerMode
from jupyter_mcp_server.notebook_manager import NotebookManager

logger = logging.getLogger(__name__)


class UnuseNotebookTool(BaseTool):
    """Tool to unuse from a notebook and release its resources"""
    
    async def execute(
        self,
        mode: ServerMode,
        server_client: Optional[JupyterServerClient] = None,
        kernel_client: Optional[Any] = None,
        contents_manager: Optional[Any] = None,
        kernel_manager: Optional[Any] = None,
        kernel_spec_manager: Optional[Any] = None,
        notebook_manager: Optional[NotebookManager] = None,
        # Tool-specific parameters
        notebook_name: str = None,
        shutdown_kernel: bool = False,
        session_manager: Optional[Any] = None,
        **kwargs
    ) -> str:
        """Execute the unuse_notebook tool.
        
        Args:
            mode: Server mode (MCP_SERVER or JUPYTER_SERVER)
            kernel_manager: Kernel manager for JUPYTER_SERVER mode (optional kernel shutdown)
            notebook_manager: Notebook manager instance
            notebook_name: Notebook identifier to disconnect
            **kwargs: Additional parameters
            
        Returns:
            Success message
        """
        if notebook_name not in notebook_manager:
            return f"Notebook '{notebook_name}' is not connected. All currently connected notebooks: {list(notebook_manager.list_all_notebooks().keys())}"
        
        # Get info about which notebook was current
        current_notebook = notebook_manager.get_current_notebook()
        was_current = current_notebook == notebook_name
        
        shut = None
        kernel_id = None
        if mode == ServerMode.JUPYTER_SERVER:
            # JUPYTER_SERVER mode: Optionally shutdown kernel before removing
            # Note: In JUPYTER_SERVER mode, kernel lifecycle is managed by kernel_manager
            # We only remove the reference in notebook_manager, the actual kernel
            # continues to run unless explicitly shutdown
            
            kernel_id = notebook_manager.get_kernel_id(notebook_name)
            if kernel_id and kernel_manager and shutdown_kernel:
                shut = await shut_down_kernel(kernel_id, kernel_manager,
                                              session_manager)
            elif kernel_id and kernel_manager:
                logger.info(f"Notebook '{notebook_name}' is being unused in JUPYTER_SERVER mode. Kernel {kernel_id} remains running.")

            success = notebook_manager.remove_notebook(notebook_name)
            
        elif mode == ServerMode.MCP_SERVER:
            # MCP_SERVER mode: Use notebook_manager's remove_notebook method
            # which handles KernelClient cleanup automatically
            success = notebook_manager.remove_notebook(notebook_name)
        else:
            return f"Invalid mode: {mode}"
        
        if success:
            message = f"Notebook '{notebook_name}' unused successfully."
            if shutdown_kernel and mode == ServerMode.JUPYTER_SERVER:
                if not (kernel_id and kernel_manager):
                    message += " It had no kernel to shut down."
                elif shut is True:
                    message += f" Its kernel {kernel_id} was shut down."
                else:
                    message += f" Its kernel {kernel_id} could not be shut down: {shut}"
            
            if was_current:
                new_current = notebook_manager.get_current_notebook()
                if new_current:
                    message += f" Current notebook switched to '{new_current}'."
                else:
                    message += " No notebooks remaining."
            
            return message
        else:
            return f"Notebook '{notebook_name}' was not found."


async def shut_down_kernel(kernel_id: str, kernel_manager,
                           session_manager=None) -> Any:
    """End ``kernel_id``'s sessions, then the kernel. True, or why not.

    The sessions first: a kernel shut down under its session left Lab showing
    the notebook attached to a kernel that no longer exists. Either manager's
    methods may be coroutines (they are in jupyter_server 2.x).
    """
    import inspect

    async def settle(value):
        return await value if inspect.isawaitable(value) else value

    try:
        if session_manager is not None:
            for session in await settle(session_manager.list_sessions()) or []:
                if (session.get("kernel") or {}).get("id") == kernel_id:
                    await settle(session_manager.delete_session(session.get("id")))
        if kernel_id in kernel_manager:
            await settle(kernel_manager.shutdown_kernel(kernel_id))
        logger.info(f"Kernel {kernel_id} shut down with its notebook")
        return True
    except Exception as e:  # noqa: BLE001 - said in the reply
        logger.warning(f"Could not shut down kernel {kernel_id}: {e}")
        return f"{type(e).__name__}: {e}"
