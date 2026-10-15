"""Offline stand-ins for the models the eight loaders call.

One per framework, each following one script: load the target skill, read its
resource, run its script, answer. Every module here imports its framework at
module level, so only a loader's child process may import one (the parent
process imports no framework at all). Nothing here asserts: grading lives in
``contract.py``.
"""
