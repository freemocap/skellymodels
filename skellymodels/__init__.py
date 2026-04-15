"""
skellymodels — Mathematically rigorous skeleton, rigid body, and trajectory
definitions for motion capture pipelines.
"""


from beartype.claw import beartype_this_package

beartype_this_package()


from skellylogs import configure_logging, LogLevels


LOG_LEVEL = LogLevels.TRACE
configure_logging(level=LOG_LEVEL)
