"""Tree tuning, verification, freezing, final evaluation, and reporting."""

from collections.abc import Sequence

__all__ = ["tuning_main"]


def tuning_main(argv: Sequence[str] | None = None) -> int:
    # Unpickling the deployment model imports tree_tuning.training, not the CLI.
    # Keep experiment reporting and resource monitoring off that import path.
    from .cli import tuning_main as main

    return main(argv)
