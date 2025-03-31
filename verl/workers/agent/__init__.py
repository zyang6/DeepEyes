# NOTE: Env must be imported here in order to trigger metaclass registering
from .envs.rag_engine.rag_engine import RAGEngineEnv
from .envs.rag_engine.rag_engine_v2 import RAGEngineEnvV2

try:
    from .envs.frozenlake.frozenlake import FrozenLakeTool
except Exception as err:
    print(f' [ERROR] Failed to register FrozenLakeTool : {err=}')

from .parallel_env import agent_rollout_loop