"""
Backward-compatible wrapper for ProjectionMemoryBank.

The original SGPMemoryBank has been refactored into ProjectionMemoryBank
in projection_memory.py, which supports both GPM and SGP methods.
This module re-exports the class under the old name for import compatibility.
"""

from learning.cl.projection_memory import ProjectionMemoryBank


class SGPMemoryBank(ProjectionMemoryBank):
    """Backward-compatible alias for ProjectionMemoryBank (defaults to GPM)."""

    def __init__(self, device, threshold=0.98):
        super().__init__(device, threshold=threshold, method="gpm")
