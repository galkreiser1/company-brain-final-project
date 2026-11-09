"""Shared test setup. Runs before any test module is imported."""

import os

# Tests use fake models and data, so their runs shouldn't land in Langfuse next to the
# real traces. Set before maya.config loads .env, which never overrides a set value.
os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
