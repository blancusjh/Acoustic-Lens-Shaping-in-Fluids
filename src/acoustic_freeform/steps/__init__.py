"""Research steps: read a declared configuration, call the library, write one new run directory.

Only modules in this package perform experiment file I/O. Each module exposes
``run(config, out)``, or ``add_arguments(parser)`` with ``cli(args)`` when the
step takes other inputs; ``acoustic_freeform.cli`` registers them.
"""
