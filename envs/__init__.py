"""SO-101 Task Environments"""

from envs.so101_base_env import SO101BaseEnv
from envs.stage_1_task import Stage1Task

# Optional stage imports (may not all exist yet)
try:
    from envs.stage_2_task import Stage2Task
except ImportError:
    Stage2Task = None

try:
    from envs.stage_2a_task import Stage2ATask
except ImportError:
    Stage2ATask = None

try:
    from envs.stage_2b_task import Stage2BTask
except ImportError:
    Stage2BTask = None

try:
    from envs.stage_2c_task import Stage2CTask
except ImportError:
    Stage2CTask = None

try:
    from envs.stage_3_task import Stage3Task
except ImportError:
    Stage3Task = None

try:
    from envs.stage_4_task import Stage4Task
except ImportError:
    Stage4Task = None

__all__ = ["SO101BaseEnv", "Stage1Task", "Stage2Task", "Stage2ATask", "Stage2BTask", "Stage2CTask", "Stage3Task", "Stage4Task"]
