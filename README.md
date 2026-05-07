<p align="center">
  <img src="logo.png" alt="Logo" width="200"/>
</p>

<h1 align="center">(Energy & Buildings) A novel reinforcement learning method based on generative adversarial network for air conditioning and energy system control in residential buildings</h1>

This repository trains and evaluates reinforcement learning and imitation learning controllers for residential air conditioning, PV, and battery operation.

## Project Structure

- `gail_control/config.py`: command line configuration.
- `gail_control/envs/`: RC environment, battery/AC dynamics, and PID baseline.
- `gail_control/training/`: Stable-Baselines3 and `imitation` orchestration.
- `gail_control/imitation/`: expert demonstration loading and heuristic expert generation.
- `gail_control/utils/`: shared utility functions.
- `scripts/`: command line entry points.
- `run_exp.py`: top-level paper-style experiment runner.
- `backup/legacy_code/`: old or currently unused code kept for reference.
- `backup/generated_cache/`: moved generated Python cache files from the previous layout.

## Required Data

By default the code expects:

- `data/simulation_data_2017_2018.csv`
- `data/simulation_data_2018_2019.csv`
- `data/normal_expert_train.csv`

Generated outputs are written under `results/`.

## Common Commands

Validate the environment:

```bash
python scripts/check_env.py
```

Run RL training with defaults:

```bash
python scripts/train.py --mode rl
```

Evaluate a trained model:

```bash
python scripts/evaluate.py --is_training false --model_path results/<setting>/best_model
```

Run imitation learning:

```bash
python scripts/train.py --mode il --expert_source heuristic
```

For control validation, `--expert_source heuristic` uses an environment-consistent
expert generated from the current RC simulator. `--expert_source csv` keeps support
for pre-generated MPC/expert demonstrations in `data/normal_expert_train.csv`.

Run the PID/SAC/GAIL comparison:

```bash
python run_exp.py --device cuda
```

Quick smoke run for the same top-level experiment:

```bash
python run_exp.py --quick --device cuda --output_dir results/run_exp_quick
```

Advanced comparison script:

```bash
python scripts/compare_control.py --device cuda
```

Run the grid experiment:

```bash
python scripts/grid_search.py
```

## Paper Alignment

The implementation follows the paper's main control setup:

- State variables: hour, indoor temperature, outdoor temperature, target temperature, occupancy schedule, battery state of charge, PV generation, total electricity consumption, and electricity price.
- Action variables: AC on/off, AC temperature setpoint, and battery charge/discharge rate.
- Direct RL reward: thermal comfort, electricity cost, and battery control terms following Eq. 13-15.
- GASAC: GAIL-style discriminator feedback with SAC generator/policy training. The current validation flow uses a heuristic expert generated inside the same RC environment, plus BC regularization for stable continuous control.
- Metrics: success rate, electricity consumption, and electricity cost are reported during evaluation.

The repository consumes pre-generated MPC expert data. It does not currently
include a full MPC optimizer or EnergyPlus co-simulation workflow for regenerating
that expert dataset from scratch.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

## Citation

If you use this repository in academic work, please cite the manuscript:

```bibtex
@article{hu2025novel,
  title={A novel reinforcement learning method based on generative adversarial network for air conditioning and energy system control in residential buildings},
  author={Hu, Zehuan and Gao, Yuan and Sun, Luning and Mae, Masayuki and Imaizumi, Taiji},
  journal={Energy and Buildings},
  volume={336},
  pages={115564},
  year={2025},
  publisher={Elsevier}
}
```