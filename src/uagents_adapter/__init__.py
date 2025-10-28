"""LlamaIndex adapter for uAgents framework."""

from importlib import metadata

# Core imports
from .common import ResponseMessage, cleanup_all_uagents, cleanup_uagent
from .uAgentsPlugin import UAgentsPlugin

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)


# Core exports
__all__ = [
    "UAgentsPlugin",
    "ResponseMessage",
    "cleanup_uagent",
    "cleanup_all_uagents",
    "__version__",
]
