# Copyright (c) 2024- Datalayer, Inc.
#
# BSD 3-Clause License

"""Jupyter MCP Server."""

# The fork's own releases: upstream 1.0.2 plus our commits. Bump the .postN
# with every commit csgui pins: pip compares versions, not commits, so a pin
# to a new commit under an unchanged version is resolved and then skipped as
# "already installed".
__version__ = "1.0.2.post1"
