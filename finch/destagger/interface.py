from . import impl
from .. import DefaultOperator as Operator
from .. import util

destagger = impl.destagger_blocked_np
"""
The default destagger implementation.

Group:
    Destagger
"""

DESTAGGER_REGEX = "destagger.*"
"""A regular expression for discovering destagger operator implementations."""


def list_implementations() -> list[Operator]:
    """Return a complete list of available implementations of the destagger operator.

    Returns:
        list[Operator]: All available implementations of the destagger operator.

    Group:
        Destagger
    """
    return util.list_funcs_matching(impl, DESTAGGER_REGEX, Operator)
