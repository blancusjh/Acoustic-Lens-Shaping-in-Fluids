"""Design a lens on the fixed machine: desired singlet in, machine settings out.

>>> from acoustic_freeform.design import Singlet, design
>>> lens = Singlet(object_m="-inf", image_m=0.06, thickness_m=0.02, shape_factor=0.0, scale=1 / 14)
>>> recipe = design(lens)          # doctest: +SKIP
>>> recipe.status, recipe.settings["fill_volumes_m3"]   # doctest: +SKIP
"""

from .machine import DEFAULT_MACHINE, Machine, Singlet
from .recipe import STATUS, Recipe, design, fill_volumes, fit_problems

__all__ = ["DEFAULT_MACHINE", "STATUS", "Machine", "Recipe", "Singlet", "design", "fill_volumes", "fit_problems"]
