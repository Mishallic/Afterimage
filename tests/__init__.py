"""The tests run against a clean environment: whatever .env says must not change their outcome."""

import os

import nohands.config  # noqa: F401  loads .env once, so the pops below stick

for _name in ("MODEL_API_KEY", "BASETEN_API_KEY", "FIREWORKS_API_KEY", "SPARK_PROVIDER"):
    os.environ.pop(_name, None)
