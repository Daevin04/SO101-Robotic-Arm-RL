# Environment Files Explained

## so101_base_env.py (BASE CLASS)

**Purpose:** Shared functionality for ALL task environments

**Does NOT contain:**
- ❌ Reward computation
- ❌ `step()` method (how actions are executed)
- ❌ Task-specific logic

**DOES contain:**
- ✅ MuJoCo model loading (`__init__`)
- ✅ Observation generation (`_get_obs`)
- ✅ Helper utilities:
  - `_get_gripper_state()` - Get gripper open/close state
  - `_has_gripper_contact()` - Check if gripper touching object
  - `_is_grasped()` - Check if object is held
  - `_is_positioned()` - Check if gripper positioned correctly
  - `_is_success()` - Check if task succeeded
- ✅ Reset logic (`reset()`)
- ✅ Rendering (`render()`, `close()`)
- ✅ Constants (joint limits, thresholds)

**Think of it as:** A toolbox that all tasks can use

---

## so101_positioning_grasp_prep_env.py (SPECIFIC TASK)

**Purpose:** Defines the GRASP PREP task (positioning + grasping + lifting)

**Inherits from:** `SO101BaseEnv` (gets all the utilities)

**Adds task-specific:**
- ✅ **Reward computation** (`_compute_reward()`) ⭐ **THIS IS THE KEY**
  - Dense rewards: Distance, centering, orientation, height
  - Sparse rewards: Contact, grasp, lift, success
- ✅ **Action execution** (`step()`)
  - How robot responds to actions
  - Physics simulation step
  - Frozen object logic
- ✅ **Action space** (6D: arm + gripper)
- ✅ **Task-specific success** (`_is_lifted()`)
- ✅ **Task initialization** (`__init__` with freeze_object)

---

## Analogy

```
so101_base_env.py = Kitchen with appliances (oven, fridge, sink)
                    ↓
so101_positioning_grasp_prep_env.py = Chef cooking a specific recipe
                                      (uses the kitchen tools)
```

**Other tasks could inherit from the same base:**
- `so101_pushing_env.py` - Different recipe (pushing task)
- `so101_closing_env.py` - Different recipe (closing task)
- All use the same kitchen (base class)!

---

## Key Difference: REWARDS

### Base Class (so101_base_env.py)
```python
# NO _compute_reward() method!
# Just provides utilities
```

### Task Class (so101_positioning_grasp_prep_env.py)
```python
def _compute_reward(self):
    """DEFINES the reward structure for THIS task"""
    # Dense rewards
    dist_reward = -5.0 * distance
    centering_reward = -2.0 * lateral_offset
    # Sparse rewards
    contact_reward = 20.0 if distance < 0.02 else 0.0
    grasp_reward = 30.0 if is_grasping else 0.0
    # ... etc
    return total_reward
```

**Each task defines its own rewards!**

---

## Visual Structure

```
SO101BaseEnv (so101_base_env.py)
    │
    ├── Load robot model
    ├── Get observations
    ├── Helper utilities (contact, grasp checks, etc.)
    ├── Reset environment
    └── Rendering
         │
         └── SO101PositioningGraspPrepEnv (so101_positioning_grasp_prep_env.py)
              │
              ├── Inherits all utilities from base
              ├── Defines GRASP PREP task
              ├── Defines REWARD STRUCTURE ⭐
              ├── Defines how actions work (step)
              └── Defines success criteria
```

---

## Summary

| Feature | Base Class | Task Class |
|---------|-----------|------------|
| **Rewards** | ❌ None | ✅ Defined in `_compute_reward()` |
| **step()** | ❌ None | ✅ Executes actions, computes rewards |
| **Utilities** | ✅ Yes | ✅ Inherits + uses |
| **Observation** | ✅ Yes | ✅ Inherits |
| **Reset** | ✅ Basic | ✅ Can override |
| **Task logic** | ❌ Generic | ✅ Task-specific |

**Bottom line:** Base class = shared tools, Task class = specific objectives + rewards
