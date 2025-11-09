# Mujoco-Robotic-Simulation

## Overview

This repository contains a collection of MuJoCo-based robot arm simulation environments and training scripts for reinforcement learning experiments. The codebase is focused on training and evaluating policies (SAC/PPO + HER and curriculum learning techniques) on a Franka/UR5-like robot model using the `mujoco` Python bindings and `stable-baselines3`.

Primary goals:
- Provide reproducible training scripts for robot manipulation tasks.
- Support Hindsight Experience Replay (HER) and simple curriculum setups.
- Offer visualization scripts and a small suite of robot XML assets used by the environments.

## Quick start

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Install MuJoCo following the official installation instructions for the `mujoco` Python package (make sure the native MuJoCo runtime and any license keys are installed/configured per the MuJoCo docs).

3. Run training or visualization scripts from the repository root. Example commands:

```bash
# Train a baseline agent
python train_robot.py

# Train with HER + curriculum (custom script)
python train_robot_HER_curriculum.py

# Train and show live visualization
python train_robot_visualize.py

# Visualize a saved model / scene
python view_panda.py

# Run the simple smoke tests
python test_robot.py
```

To inspect training logs with TensorBoard:

```bash
tensorboard --logdir tensorboard_logs
```

## Repository structure (key files and folders)

- `train_robot.py` — Main training entrypoint (baseline RL training using stable-baselines3).
- `train_robot_HER_curriculum.py` — Training script that includes a HER + simple curriculum learning setup.
- `train_robot_visualize.py` — Training script with live visualization enabled.
- `train_robot.py` / `train_robot_HER_curriculum.py` may use different hyperparameters and wrappers — read their headers for details.
- `view_panda.py` — Simple viewer for the Panda robot XML scenes.
- `test_robot.py` — Basic tests and smoke-run utilities for quick checks.
- `*.xml` (e.g., `robot_arm.xml`, `ur5e.xml`, `objects.xml`) — MuJoCo model and scene definitions used by the environments.
- `franka_emika_panda/` — Folder containing Franka/Panda-specific MuJoCo XML assets and meshes.
- `checkpoints/` — Saved model weights and checkpoints produced by training runs.
- `tensorboard_logs/`, `panda_tensorboard/`, `panda_her_tensorboard/` — TensorBoard event logs and organized runs for experiment tracking.
- `requirements.txt` — Python dependencies used by the project.

## Details & notes

- This project uses `mujoco` (the upstream Python bindings) and `stable-baselines3` for RL algorithms. The `requirements.txt` pins minimal versions that were used during development.
- MuJoCo requires a working native runtime and (depending on your MuJoCo distribution) a license or activation step. Follow the official MuJoCo install instructions for your platform (Windows in this workspace).
- The training scripts save checkpoints into the `checkpoints/` folder and log TensorBoard data under `tensorboard_logs/` or the per-experiment folders shown in the repository.
- Many XML robot assets are included under `franka_emika_panda/` and the top-level XML files in the repo. You can modify these to change physics parameters, link sizes, or sensors.

## How to run experiments (tips)

- Use a virtual environment to isolate dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # or use your platform-appropriate activate script
pip install -r requirements.txt
```

- If you run into GPU/renderer issues on Windows, try running from the bundled `bash.exe` shell (WSL or Git Bash) if available, or adjust the MuJoCo/OpenGL settings per the MuJoCo docs.
- To reproduce an experiment, note the `tensorboard` folder name and the checkpoint path. The training scripts typically print where they save models.

## Troubleshooting

- Import errors: verify that `mujoco` is installed and that your Python environment is the same one used to install the package.
- Renderer errors (OpenGL): ensure your OS has appropriate GL drivers or use a headless/back-end renderer if supported.
- Licensing/activation: if MuJoCo refuses to start, confirm your MuJoCo license/activation steps are completed.

## Contributing

This repository appears to be a personal/academic project. If you plan to contribute, please:

1. Fork and create a feature branch.
2. Run and validate tests (`python test_robot.py`).
3. Open a PR describing the change and any experimental results.

## License

Check individual files for license headers. The `franka_emika_panda/` folder includes asset-level license notes (see `franka_emika_panda/README.md`). This repository does not currently include a top-level license file; add one if you intend to publish or share.

## Contact / Author

This workspace belongs to the local user (see repository owner). For questions about running or extending experiments, inspect the training scripts and XML assets in this repo.

---

If you'd like, I can also:
- Add example commands for loading a specific checkpoint into `view_panda.py`.
- Add a short `CONTRIBUTING.md` and a top-level `LICENSE` (if you tell me which license you prefer).
