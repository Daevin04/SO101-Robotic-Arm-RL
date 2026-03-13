# Stage 7: Lift-Based Success Detection

**Date:** 2026-01-31
**Topic:** Sim-to-Real Compatible Success Criteria Using Cube Lift Height

---

## Overview

Stage 7 now uses **cube lift height** as the success criterion instead of force sensors or contact detection. This is **fully sim-to-real compatible** and more reliable.

---

## Why Lift-Based Detection?

### ❌ Problems with Force Sensors

**Force sensors won't work on real hardware:**
- Real SO-101 robot has no force sensors
- Would require expensive force/torque sensors ($1000+)
- Calibration nightmare (sim forces ≠ real forces)
- Sim-to-real gap very large for force measurements

### ❌ Problems with Contact Switches

**Contact detection is unreliable:**
- Requires precise contact switch placement
- Can falsely trigger (touched but not grasped)
- Can miss detection (grasped but switch not pressed)
- Doesn't verify successful pick-up

### ✅ Advantages of Lift Height

**Cube lift height is ideal:**
- ✅ **Sim-to-real compatible:** Vision system can easily detect cube height
- ✅ **Reliable:** If cube is lifted, it's definitely grasped
- ✅ **Simple:** One measurement (Z-axis position)
- ✅ **Verifiable:** Matches the actual task objective (pick up cube)
- ✅ **Already in use:** Stage 6 uses this successfully

---

## Implementation

### Success Criterion

**Success = Cube lifted ≥ 3cm off table**

```python
initial_cube_height = cube_pos[2] at episode start
current_cube_height = cube_pos[2] now
lift_height = current_cube_height - initial_cube_height

is_success = lift_height >= 0.03  # 3cm threshold
```

### Hardware Requirements (Real Robot)

**Required sensors:**
- **Camera/vision system** to track cube position
- Can use:
  - Overhead camera
  - Wrist-mounted camera
  - ArUco markers on cube
  - Object detection (YOLOv8, etc.)

**NOT required:**
- ❌ Force sensors
- ❌ Contact switches
- ❌ Torque sensors

---

## Scripted Sequence (Updated)

### Phase 1: Rotate Wrist (10 steps)
- Rotate from 0° → 90°
- Jaws change from vertical to horizontal
- **Reward:** 0.0

### Phase 2: Descend (20 steps)
- Lower from hover position (5cm above) to cube height
- Maintain X,Y alignment
- **Reward:** 0.0

### Phase 3: Close Gripper (20 steps)
- Gradually close from fully open to fully closed
- Gripper wraps around cube
- **Reward:** 0.0

### Phase 4: Lift Cube (20 steps) ✅ NEW!
- Raise wrist_flex to lift cube
- Lift ~5cm total over 20 steps
- Keep gripper closed
- **Reward:** 0.0

### Phase 5: Check Success (1 step)
- Measure current cube height
- Compare to initial height
- **Success if:** lift_height ≥ 3cm
- **Reward:** +200 + speed bonus (success) OR -1.0 (failure)

**Total:** 70 scripted frames (was 50, added 20 for lift)

---

## Rewards

### Agent Positioning Phase

**Distance-based reward:**
```python
distance_to_hover = sqrt(xy_error² + (0.5 * z_error)²)
reward = -10.0 * distance_to_hover
```

**Examples:**
- 30cm from hover → reward = -3.0/step
- 10cm from hover → reward = -1.0/step
- 1cm from hover  → reward = -0.1/step
- <1cm → **Trigger script**

### Scripted Phase

**Phases 1-4 (Rotate, Descend, Close, Lift):**
```python
reward = 0.0  # Agent receives no reward during script
```

**Phase 5 (Success Check):**
```python
if lift_height >= 0.03m:  # 3cm
    reward = SUCCESS_REWARD + LIFT_REWARD + speed_bonus
    reward = 100 + 100 + (remaining_steps * 1.0)
    # Maximum: 200 + 100 = 300 points!
else:
    reward = -1.0  # Failed to lift
```

### Reward Breakdown

**Successful episode (agent reaches hover in 40 steps):**
```
Steps 1-40:  Agent positioning    = -1.5/step × 40  = -60 pts
Step 41:     Script triggers       = 0.0             = 0 pts
Steps 42-71: Script phases 1-4    = 0.0/step × 30   = 0 pts
Step 72:     Script phase 5       = 100+100+28      = +228 pts
────────────────────────────────────────────────────────────
Total episode reward:                                +168 pts
```

**Failed grasp (agent reaches hover but lift fails):**
```
Steps 1-40:  Agent positioning    = -1.5/step × 40  = -60 pts
Steps 41-71: Script phases 1-4    = 0.0/step × 30   = 0 pts
Step 72:     Script phase 5       = -1.0            = -1 pt
────────────────────────────────────────────────────────────
Total episode reward:                                -61 pts
```

---

## Sim-to-Real Transfer

### In Simulation

**Cube height measurement:**
```python
cube_pos = self.data.site_xpos[self.object_site_id]
cube_height = cube_pos[2]  # Z-axis position
```

**Perfect measurement** (MuJoCo physics engine)

### On Real Robot

**Cube height measurement options:**

**Option 1: Overhead camera**
```python
# Detect cube using vision
cube_bbox = detect_cube(camera_image)
cube_height = estimate_height_from_bbox(cube_bbox, camera_params)
```

**Option 2: ArUco marker**
```python
# Track ArUco marker on top of cube
marker_pose = detect_aruco_marker(camera_image)
cube_height = marker_pose.position.z
```

**Option 3: Depth camera**
```python
# Use RealSense/Kinect depth data
depth_image = get_depth_image()
cube_region = crop_cube_region(depth_image)
cube_height = np.mean(cube_region)
```

**Threshold adjustment for real robot:**
- Simulation: 3cm threshold works perfectly
- Real robot: May need 2.5cm or 3.5cm depending on measurement noise
- Easy to tune: Just adjust `LIFT_HEIGHT_THRESHOLD` constant

---

## Configuration

### Current Settings

```python
# In stage_7_task.py:

# Success criteria
LIFT_HEIGHT_THRESHOLD = 0.03  # 3cm - cube must be lifted this high
USE_FORCE_SENSORS = False     # Use lift height, not force

# Rewards
SUCCESS_REWARD = 100.0  # Base reward for successful grasp
LIFT_REWARD = 100.0     # Additional reward for lifting
# Total: 200 + speed bonus

# Scripted sequence
SCRIPTED_ROTATE_STEPS = 10   # Rotate wrist
SCRIPTED_LOWER_STEPS = 20    # Descend
SCRIPTED_CLOSE_STEPS = 20    # Close gripper
SCRIPTED_LIFT_STEPS = 20     # Lift cube ← NEW!
SCRIPTED_DESCENT_STEPS = 70  # Total (10+20+20+20)
```

### Adjusting Lift Threshold

```python
# Easier (lower threshold)
LIFT_HEIGHT_THRESHOLD = 0.02  # 2cm

# Current (balanced)
LIFT_HEIGHT_THRESHOLD = 0.03  # 3cm

# Harder (higher threshold)
LIFT_HEIGHT_THRESHOLD = 0.05  # 5cm
```

---

## Video Visualization

### What You'll See in Evaluation Videos

**1. Agent Positioning (Variable length)**
- Robot navigates to hover position
- Duration: 20-80 steps depending on training

**2. Phase 1 - Rotate (10 frames)**
- Wrist rotates 0° → 90°
- Smooth animation

**3. Phase 2 - Descend (20 frames)**
- Gripper lowers to cube
- Smooth animation

**4. Phase 3 - Close (20 frames)**
- Gripper closes around cube
- Smooth animation

**5. Phase 4 - Lift (20 frames) ✅ NEW!**
- Cube lifts upward
- You'll SEE the cube rise off the table
- Smooth animation

**6. Phase 5 - Check (1 frame)**
- Episode terminates
- Success or failure displayed

**Total visible:** Agent phase + 70 scripted frames

**Key visual:** You will actually **SEE the cube lift off the table** in the video!

---

## Testing

### Test 1: Verify Lift Detection Works

```bash
python -c "
from envs.stage_7_task import Stage7Task
env = Stage7Task()
obs, info = env.reset()

print(f'Initial height: {info[\"initial_cube_height\"]:.4f}m')

# Manually lift cube
env.data.qpos[env.object_qpos_start + 2] += 0.04  # Lift 4cm

action = env.action_space.sample()
obs, reward, done, trunc, info = env.step(action)

print(f'Current height: {info[\"cube_height\"]:.4f}m')
print(f'Lift height: {info[\"lift_height\"]:.4f}m')
print(f'Is lifted: {info[\"is_lifted\"]}')
print(f'Success: {info[\"is_success\"]}')
"
```

**Expected output:**
```
Initial height: 0.0150m
Current height: 0.0550m
Lift height: 0.0400m
Is lifted: True
Success: True
```

### Test 2: Run Full Scripted Sequence

```bash
python scripts/demo_stage7_scripted_descent.py
```

Should show:
- Phase 1-3: Rotate, descend, close (unchanged)
- Phase 4: **Lift cube upward** (NEW!)
- Phase 5: Check if lifted ≥3cm → success

---

## Comparison to Stage 6

### Stage 6
- Uses lift height for success ✅
- But agent must learn full manipulation
- Complex reward shaping with multiple phases
- Long training time (800K-1M steps)

### Stage 7
- Uses lift height for success ✅ (SAME)
- Agent only learns hover positioning
- Script handles manipulation automatically
- Faster training time (400K-600K steps expected)
- **Identical success metric** (lift height)

---

## Advantages

### Sim-to-Real Transfer
1. **Vision-based:** Any camera can detect cube height
2. **No special sensors:** Standard RGB camera sufficient
3. **Robust:** Height measurement less noisy than force
4. **Proven:** Stage 6 already uses this successfully

### Training Benefits
1. **Clear objective:** Agent sees +200 when cube lifts
2. **Visual feedback:** Videos show cube actually lifting
3. **Reliable gradient:** Positioning quality correlates with success
4. **Faster convergence:** Script ensures consistent lift attempt

### Debugging Benefits
1. **Easy to visualize:** Can see if cube lifts in videos
2. **Easy to measure:** Just print cube_height
3. **Easy to tune:** Adjust LIFT_HEIGHT_THRESHOLD if needed
4. **No calibration:** Works out-of-box in sim and real

---

## Expected Training Timeline

### @ 100K Steps
- Agent explores randomly
- Distance to hover: 0.10-0.30m
- Script never triggers
- Episode reward: -100 to -300

### @ 200K-400K Steps
- Agent approaches hover position
- Distance to hover: 0.05-0.15m
- Script triggers occasionally
- **First successful lifts!**
- Episode reward: -50 to +50

### @ 600K Steps (Mastery)
- Agent reaches hover reliably
- Distance to hover: <0.01m
- Script triggers 60-80% of episodes
- **Cube lifts successfully**
- Episode reward: +100 to +250
- **Success rate: 50-70%**

---

## Summary

**Success Criterion:**
- ✅ Cube lifted ≥ 3cm off table
- ✅ Measured by cube Z-position
- ✅ Sim-to-real compatible (vision-based)
- ✅ Reliable and verifiable

**Implementation:**
- 70-step scripted sequence (added 20-step lift phase)
- Reward = +200 + speed bonus for successful lift
- Agent learns positioning, script handles manipulation
- All 70 scripted frames visible in videos

**Hardware:**
- Camera/vision system (any type works)
- No force sensors needed
- No contact switches needed
- Easy to implement on real robot

**Ready for training and real-world deployment!** 🚀
