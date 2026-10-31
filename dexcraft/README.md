# dexcraft — the environment package

```
dexcraft/
  __init__.py     registers every task as a gymnasium id, `DexCraft/<task>-v1`
  env.py          DexCraftEnv (gymnasium): observation groups, control modes, ink, rewards, perturbation and
                  constraint presets, L5 paint (wells, brush rack, colour), L6 pick & place
  scene.py        UR5e + dex_hand hand (dex_hand_ur5e.xml), the tool (pen / brush / chopsticks), paper with a
                  paintable ink texture, cameras; L6 adds the fixed lower stick, the object and the bowl
  tasks.py        the task families in levels L0-L6 (plus evaluation-only E2 tasks); every task carries a
                  per-point target press force
  fonts.py        single-line block capitals and Hershey faces, digits, Chinese stroke medians, shapes, cursive "daxo"
  gen_pen.py      tools: ballpoint pen (140 x 9 mm, rubber grip, 0.6 mm ball tip), calligraphy brush (bamboo handle,
                  ferrule, 35 mm tuft - drawn only, its mechanics live in brush.py) and a 230 mm chopstick as revolved
                  meshes (assets/*.stl)
  brush.py        the brush tuft as a reduced-order footprint model outside MuJoCo (ellipse with lag, hardening normal
                  force, ink reservoir), the footprint ink renderer, the oracle reference render, the brush metrics
                  (stroke IoU, skeleton Chamfer, ink SSIM, stroke order, violations) and its acceptance tests
  grip.py         solves the tripod grip (geometric CEM -> physics-in-the-loop servo refinement with static +
                  lateral-load tests -> settled state), cached in grip_pocket.json
  planning.py     reference decoder: task map -> ordered strokes + target force (optional to use)
  metrics.py      per-episode metrics (completion, precision, force error, time, ink IoU, finger / wrist / arm share,
                  brush and paint metrics) and their aggregation
  benchmark.py    the evaluation contract (env settings every reported number uses) and per-level bars
  evaluate.py     run a policy over tasks x seeds x tracks -> results json (+ snapshots)
  record.py       record demonstrations from any teacher policy into sharded .npz
  bc.py           behaviour cloning from recorded shards into a train_ppo checkpoint
  train_ppo.py    PPO trainer (warm-starts from a BC checkpoint with --init)
  dataset.py      export the task content (stroke vectors, target images, the policy's map input) for inspection
  demo.py         watch a policy write in the viewer, or export a GIF
  view.py         open the scene in the MuJoCo viewer (joint sliders, contacts, live policy playback)
  selftest.py     end-to-end install check
  policies/       base.Policy and load_policy, scripted.ScriptedTracker (IK teacher), ppo.PPOPolicy,
                  hybrid (IK arm + learned grip), router (per-task experts), remote.RemotePolicy + serve.py (HTTP)
```

## Quick start

```bash
pip install -e ".[train]"          # from the repository root; editable, the hand model lives beside the package
python -m dexcraft.selftest        # ~30 s; checks deps, env, decoder, baseline, trainer

# the env
python -c "import gymnasium as gym, dexcraft; e=gym.make('DexCraft/L2_write_word-v1'); print(e.reset(seed=0)[0]['instruction'])"

# score a policy (module:factory or path/to/policy.py exposing make_policy())
python -m dexcraft.evaluate --policy dexcraft.policies.scripted:make_policy --seeds 3 --out results/scripted.json

# look at the scene / watch it write (macOS: mjpython)
mjpython -m dexcraft.view
mjpython -m dexcraft.view --tool chopsticks
mjpython -m dexcraft.demo --task L2_write_word --view

# demonstrations, generated from any teacher
python -m dexcraft.record --tasks write --episodes 200 --out data/scripted

# re-solve and cache the grip (only after changing the hand, tool or contacts; ~5 min)
python -m dexcraft.grip
```

The training and evaluation pipeline used for the paper is in `scripts/` at the repository root.

## Observation and action

A closed-loop controller called at `control_hz` (25 Hz recommended): `act(obs) -> action`. Every policy
sees the same four groups; how it encodes them is up to the model. `dexcraft.selftest` asserts the
dimensions.

| key | content | shape | on a real rig |
|---|---|---|---|
| `obs["position"]` | joint angles (34), joint velocities ×0.1 (34), arm joint servo targets (7). Joint order: 7 arm (6 UR5e + forearm roll), 2 wrist, 25 hand | (75,) | joint encoders |
| `obs["tactile"]` | per-joint (motor + applied + contact reaction) torque as a fraction of that joint's limit. The tool–sheet contact force is not here; it is privileged and only in `info` | (34,) | joint torque sensing |
| `obs["task"]` | **the question**: `map` is the sheet itself, 400 × 300 mm at 1 mm/px, with the target drawn in sheet coordinates; its *darkness* is the target press force (255 = `FORCE_MAX`) and its thickness the ink width that force implies. `vec`: time fraction, elapsed time, tool one-hot, `FORCE_MAX`, the map's half-size and centre | map (300,400) uint8<br>vec (10,) | the task sheet |
| `obs["external"]` | `map`: the ink on the sheet so far, same geometry. `vec`: hand pose, tool / tip / sheet / object poses in the hand frame, the stroke schedule (`write_target_in_hand`, `write_ahead_near/far`, `write_pen_down`) and the brush state (7) | map (300,400) uint8<br>vec (48,) | camera + tracking |
| `obs["instruction"]` | the task in words | str | |
| `obs["images"]` | RGB from the configured cameras (default `top`, `wrist`; `side` / `front` available) when `obs_mode` includes `"pixels"` | {cam: (224,224,3)} | cameras |

The `external` group is part of the standard observation. With `obs_mode` `"deployable"` a policy runs
without it (position + tactile + task), which is the harder setting.

**Action — arm position + hand torque**, 34 floats in [-1, 1] (control mode `"hybrid"`, the one the
benchmark scores):

* `a[0:7]` — the arm joints: a step on the joint **position** target, `target += a · max_vel · dt`.
  That is what `servoj` takes on a real UR5e, which does not accept joint torque.
* `a[7:34]` — the 2 wrist + 25 hand joints: motor **torque**, as a fraction of what that joint can
  deliver **at its current speed** — a duty cycle in the same sense as PWM. Available torque falls
  linearly to zero at `NO_LOAD_SPEED × max_vel` (1.5× rated speed).
* The hand has no position setpoint. A policy that wants position control closes its own loop —
  `policies/scripted.py` plans in joint space and runs a PD around that plan.
* A policy may also return an action chunk `(K, 34)`; the rows are applied in order.

The other control modes (`pos`, `delta_pos`, `vel`, `acc`, `torque`) drive every joint the same way.

`env.unpack(obs[group], group)` gives a named view of a group. `planning.map_to_strokes` is a reference
decoder from the task map to ordered strokes with a target force per point — use it, replace it, or
plan another way. A policy is a Python module exposing `make_policy() -> Policy` (see
`policies/base.py`), or a running policy server (`policies/serve.py` / `remote.py`).

Every episode re-randomises position, scale, slant, rotation and the force profile from the seed, and
`evaluate --fresh-seeds` draws a new seed base, so the continuous part of a task cannot be memorised.
The `E2_*` tasks (unseen words, idioms, unseen Chinese characters) are evaluation-only and never part of
a training sweep.

## Environment

```python
from dexcraft.env import DexCraftEnv
env = DexCraftEnv("L2_write_word", control_mode="hybrid", control_hz=25)   # obs_mode="pixels+state" adds images
obs, info = env.reset(seed=0)
obs, r, term, trunc, info = env.step(action)
```

* **Arm**: Universal Robots UR5e (MuJoCo Menagerie, BSD-3) mounted behind the sheet, with the hand on its
  flange. The arm joints get gravity compensation, as the real controller does.
* **Reset** restores a cached, physically settled grip (tool held by friction, wrist pitched so the pen
  meets the paper at a writing angle).
* **Tools**: L0–L2 and L4 use the ballpoint pen; L3 and L5 use the brush (L5 paints in colour: paint wells
  and a three-brush rack beside the sheet); L6 uses chopsticks. The brush is two layers (`brush.py`): the
  handle and ferrule are rigid bodies the fingers hold; the tuft is a reduced-order footprint model
  outside MuJoCo — an ellipse (a, b, φ) that lags the shaft's depth and tilt, a hardening normal force
  k·d^α applied back on the shaft, ink-dependent friction and an ink reservoir that only runs down.
  Presets live in `assets/brush_configs.json`; every value is uncalibrated and the tier is
  simulation-only. `python -m dexcraft.brush` runs its acceptance tests. All tools share one body-frame
  convention (grip point 42 mm below the origin), so one solved grip serves all of them; the tip-heavy
  chopstick gets its own servo refinement (`servo_pose_chopsticks`).
* **Chopsticks (L6)**: the upper stick is the free tool body (held like the pen, 150 mm above the tip);
  the lower stick is fixed to `hand_base`, as it is wedged in the thumb web and resting on the ring
  finger in a proper grip. Closing is a finger motion (flex index + middle). The object sits on the
  sheet and the bowl is a mocap body.
* **Ink** is painted into the paper texture from actual tool–paper contact, with a width that grows with
  the press force. The brush stamps its footprint ellipse instead, breaking up when the ink runs low.
* **Perturbation presets** (`mild`, `hard`): tool mass / friction, paper friction, random pushes,
  observation noise, action delay, light / camera jitter. **Constraint presets**, e.g. `arm_lock`
  (finger track: the arm is frozen whenever the pen touches the paper), `slow_arm`, `fingers_only`,
  `idle_lock` (ring and pinky servoed to the grip pose).
* **Finger / wrist share** (`finger_share`, `wrist_share`, `arm_share` in the metrics): the fraction of
  pen-tip motion while writing produced by each joint group (Jacobian split of the tip velocity).

## Evaluation

`evaluate.py` runs three tracks over the same task seeds: `nominal` (no perturbation), `robust` (perturb
preset `mild`) and `constrained` (preset `arm_lock`). The results json holds `per_task`, `summary` and
`score`; the score axes (completion, quality, time, robustness, constrained) are reported separately,
with no weighted total. `benchmark.py` defines the env settings every reported number uses; any
`--env-kwargs` override of them is recorded in the results json as a contract deviation.

## Notes on the physics

* `noslip_iterations=5` is required: without it the held pen creeps under static friction.
* The pen is a ballpoint shape (`gen_pen.py`): collision = body and rubber-grip capsules, a cone capsule
  and a 0.6 mm ball tip; visual = revolved meshes.
* A thin pen pivots easily about the fingertips under the tip's lateral friction, so the grip solver adds
  a 4th contact (shaft on the index base / web) and optimises the servo targets with a 0.7 N lateral-load
  test at the tip (`grip.refine_servo`).
* Pressing is part of the task: a policy must regulate contact force or it pushes the pen out of the grip.
