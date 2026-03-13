# How the Grasped Flag Works

## Overview

The **grasped flag** is a binary signal (0.0 or 1.0) in the observation that tells the agent: "You are currently holding the object."

**Location in observation:** Position 29 (last element of 30D observation vector)

---

## The Logic Chain

### 1. In Observation (Line 114 in so101_base_env.py)

```python
grasped = np.array([1.0 if self._is_grasped() else 0.0])
#                          ↑
#                          Calls helper function
```

**Output:**
- `1.0` = "Yes, you're grasping the object"
- `0.0` = "No, you're not grasping"

---

### 2. Grasped Check (Line 151-156)

```python
def _is_grasped(self):
    """Check if object is grasped (closed gripper + contact)."""
    gripper_state = self._get_gripper_state()        # Get gripper position
    has_contact = self._has_gripper_contact()        # Check for contact
    gripper_closed = gripper_state < 0.25            # Is gripper closed?
    return has_contact and gripper_closed            # Both must be true
    #      ↑               ↑
    #   PRIVILEGED      REALISTIC
```

**Returns `True` only if BOTH:**
- ✅ Gripper is closed (gripper_state < 0.25)
- ✅ Contact detected between gripper and object

---

### 3. Contact Detection (Line 137-149) ⚠️ PRIVILEGED

```python
def _has_gripper_contact(self):
    """Check if gripper is in contact with object."""
    for i in range(self.data.ncon):  # Loop through ALL contacts in simulation
        contact = self.data.contact[i]
        geom1 = contact.geom1  # First geometry in contact
        geom2 = contact.geom2  # Second geometry in contact

        # Check if contact involves object
        if (geom1 == self.object_geom_id or geom2 == self.object_geom_id):
            # AND involves gripper or jaw
            if (geom_bodyid[geom1] in [gripper_body_id, jaw_body_id] or
                geom_bodyid[geom2] in [gripper_body_id, jaw_body_id]):
                return True  # Collision detected!

    return False  # No collision
```

**How it works:**
1. MuJoCo maintains list of ALL collisions: `self.data.ncon`
2. Loop through each collision pair
3. Check if collision is between:
   - Object geometry (`object_geom_id`)
   - AND gripper/jaw body
4. If yes → contact detected!

**This is PRIVILEGED because:**
- Uses MuJoCo's perfect physics collision detection
- Real robot doesn't have this - needs sensors!

---

## Visual Flow

```
Agent action → Gripper closes
                    ↓
            Gripper touches object
                    ↓
    MuJoCo detects collision (PRIVILEGED!)
                    ↓
         _has_gripper_contact() = True
                    ↓
         gripper_state < 0.25? Check
                    ↓
              Both true?
                    ↓
           _is_grasped() = True
                    ↓
         Grasped flag = 1.0 (in obs)
                    ↓
         Agent sees: "I'm holding it!"
```

---

## Why It's Privileged

### In Simulation (Current):
```python
# MuJoCo tells us EXACTLY when collision happens
self.data.ncon  # Number of contacts
self.data.contact[i]  # Contact details (which geometries, forces, etc.)
```

### On Real Robot:
You would need ONE of these:

**Option 1: Force/Torque Sensor**
- Mounted at wrist
- Detects forces when object contacts gripper
- If force > threshold → contact = True

**Option 2: Tactile Sensors**
- On gripper fingers
- Binary touch sensors
- If sensor pressed → contact = True

**Option 3: Motor Current Sensing**
- Monitor gripper motor current
- If current spikes (resistance) → contact = True

**Option 4: Vision-Based**
- Camera sees object between gripper fingers
- Image processing detects grasp
- Computationally expensive

**Option 5: Inference (Least Reliable)**
- If gripper closed + object not falling → grasped
- No direct sensing
- Can be wrong!

---

## What Agent Learns From This

The grasped flag helps the agent learn:

1. **"I'm holding it now"** - Immediate feedback
2. **Associate:** Closed gripper + proximity → grasped flag = 1.0
3. **Reward correlation:** Grasped flag = 1.0 → big rewards (+30, +50, +100)
4. **Policy learns:** "Close gripper when near object → grasped → rewards"

**Without grasped flag:**
- Agent would need to infer from object movement
- Harder to learn
- Slower training

---

## In the Reward Function

The grasped flag is NOT directly used in rewards in `so101_positioning_grasp_prep_env.py`.

**Instead, rewards recompute contact each step:**

```python
# Line 207-210 in _compute_reward()
is_grasping = (distance < self.CONTACT_DISTANCE and
               gripper_normalized < self.GRASP_THRESHOLD)
if is_grasping:
    grasp_reward = 30.0
```

**But wait - this uses distance, not contact!**

Actually checking further in the code... let me verify if it actually uses the contact detection or just distance approximation.

---

## Observation vs Reward

**In OBSERVATION (line 114):**
```python
grasped = np.array([1.0 if self._is_grasped() else 0.0])
# Uses _has_gripper_contact() → PRIVILEGED contact detection
```

**In REWARD (line 207-210):**
```python
is_grasping = (distance < 0.02 and gripper_normalized < 0.5)
# Uses distance approximation, NOT actual contact detection
# distance < 2cm = "close enough to be contact"
```

**So:**
- Observation has TRUE contact via collision detection (privileged)
- Reward uses DISTANCE APPROXIMATION (could work on real robot)

This is actually smart! Rewards don't rely on privileged contact, but agent can use the privileged grasped flag to learn faster.

---

## Summary

**Grasped flag = Binary signal in observation (position 29)**

**How it works:**
1. Check if gripper is closed (< 0.25 normalized position)
2. Check if MuJoCo detects collision between gripper and object
3. If BOTH true → grasped = 1.0

**Privileged because:**
- Uses MuJoCo's perfect collision detection
- Real robot needs force/tactile sensors

**What agent learns:**
- "When this flag = 1.0, I'm holding the object"
- Helps agent understand when grasp succeeded
- Correlates with big rewards

**For real robot:**
- Need force/torque sensor to replace contact detection
- OR train vision-based policy that doesn't use this
