# Walkthrough: Train an eSEN/UMA model on an omol subset with solvent conditioning (SLURM)

## Context

The user is on the `esen-custom-params` branch of FAIRChem, which adds optional **solvent conditioning** to the UMA/eSCN-MD ("eSEN") backbone. We've already enabled this feature in their configs:

- `configs/uma/training_release/backbone/K4L2.yaml` — `use_solvent_embedding: True`, `solvent_emb_grad: True`, `solvent_emb_hidden: 16`
- `configs/uma/training_release/dataset/{uma,uma_finetune,uma_debug}.yaml` — `'solvent'` added to `r_data_keys`

The user now wants to take **raw quantum-chemistry outputs (ORCA / xTB) for a subset of omol-like molecules** and train a UMA-small model on **SLURM**. Solvent conditioning widens `mix_csd`; this can be trained from scratch, or grafted onto a pretrained UMA checkpoint when finetuning (the pretrained `mix_csd` weight is zero-padded into the new solvent block at load time — see `configs/uma/finetune/uma_sm_solvent_finetune.yaml`). The walkthrough below covers the from-scratch path.

The end-to-end pipeline has five stages: **parse → ASE DB → normalize/element-refs → configs → SLURM submit**.

---

## Step 1 — Parse QC outputs into ASE Atoms (user code)

ASE has built-in parsers, but coverage of ORCA/xTB is partial. Practical pattern:

```python
from ase.io import read
from ase.calculators.singlepoint import SinglePointCalculator

atoms_list = []
for orca_out in glob.glob("orca_runs/*/*.out"):
    atoms = read(orca_out)  # geometry
    energy = parse_orca_energy(orca_out)  # user helper
    forces = parse_orca_forces(orca_out)  # user helper
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    atoms.info["charge"] = 0
    atoms.info["spin"] = 0
    atoms.info["solvent"] = "water"  # or list_solvents() name; "" / None = vacuum
    atoms_list.append(atoms)
```

**Required atoms attributes**, read by `AtomicData.from_ase()` in `src/fairchem/core/datasets/atomic_data.py:341-537`:
- `positions`, `atomic_numbers`, `cell`, `pbc` — always
- Energy + forces — via `atoms.calc.results` (preferred) or `atoms.info["energy"]/["forces"]`
- `atoms.info["charge"]`, `atoms.info["spin"]`, `atoms.info["solvent"]` — gated by `r_data_keys`

**Solvent labels**: must be a name from `fairchem.core.datasets.solvent.list_solvents()` (179 entries from the Minnesota Solvent Descriptor DB), or one of `{"", "vacuum", "gas", "gas_phase", "gas-phase", "none"}` for vacuum. Unknown names with `strict=False` (the `from_ase` default) log a warning and fall back to the vacuum vector — verify your names ahead of time.

**Caveat**: `ase.io.read` for ORCA reads the geometry but doesn't reliably extract energies/forces from all ORCA versions, and xTB output parsing varies. Expect to write small parsers; the rest of the pipeline doesn't care how you got the values.

---

## Step 2 — Write an ASE DB and compute element references + normalizer

Use the existing helper script (`src/fairchem/core/scripts/create_finetune_dataset.py`) as the template. It:

1. Writes ASE Atoms to `data.NNNN.aselmdb` via `ase.db.connect(...).write(atoms, data=atoms.info)` (lines 97-124).
2. Computes the per-element energy reference (`compute_normalizer_and_linear_reference`, line 28+) — needed for `element_refs`.
3. Computes the force RMS — used for `normalizer_rmsd` in the top YAML.

**Recommended layout** under your `data_root_dir`:
```
my_data/
├── omol_subset/
│   ├── train/        data.0000.aselmdb, data.0001.aselmdb, ...
│   └── val/          data.0000.aselmdb
├── element_refs.npz  # output of the normalizer script
└── force_rmsd.txt    # output of the normalizer script
```

Splitting train/val is your responsibility — random 95/5 by molecule (not by snapshot) is the usual choice.

---

## Step 3 — Write a custom dataset config (single dataset only)

The default `dataset/uma.yaml` concatenates five datasets (oc20/omol/omat/odac/omc). You only have omol — copy and trim.

Create `configs/uma/training_release/dataset/omol_subset.yaml`:

```yaml
omol_final_snapshot_dir: omol_subset           # relative to cluster.data_root_dir

omol_train:
  splits:
    train:
      src: ${cluster.data_root_dir}/${dataset.omol_final_snapshot_dir}/train
  format: ase_db
  a2g_args:
    molecule_cell_size: 120.0                  # auto-cell for molecules
    r_energy: True
    r_forces: True
    r_data_keys: ['spin', 'charge', 'solvent']  # solvent enabled
    r_edges: ${cpu_graph}
    radius: ${cutoff_radius}
    max_neigh: ${max_neighbors}
  key_mapping:
    energy: omol_energy
    forces: ${omol_forces_key}
  transforms:
    common_transform:
      dataset_name: omol

omol_val:
  splits:
    val:
      src: ${cluster.data_root_dir}/${dataset.omol_final_snapshot_dir}/val
  format: ase_db
  a2g_args: { ... }    # same as above
  key_mapping: { ... }
  transforms: { common_transform: { dataset_name: omol } }
```

The `dataset_name: omol` tag is what feeds the dataset embedding inside the backbone.

---

## Step 4 — Write a custom tasks config (omol only)

The default `tasks/uma_direct.yaml` defines five energy tasks. With only omol, drop the other four. Create `configs/uma/training_release/tasks/omol_only.yaml` containing just the `omol_energy` task and the `forces` task, with `datasets: [omol]` on forces. This avoids the trainer trying to mask losses for datasets that aren't loaded.

(`MTCollater` fills missing targets with `inf` for loss masking per CLAUDE.md, so leaving extra tasks would technically work, but a clean tasks file is easier to reason about.)

---

## Step 5 — Write a SLURM cluster config

Create `configs/uma/training_release/cluster/my_slurm.yaml`, modeled on `cluster/h100.yaml`:

```yaml
data_root_dir: /path/to/my_data          # absolute path on shared filesystem
run_dir: /path/to/outputs/uma_solvent    # checkpoints + logs land here
account: <your_slurm_account>
qos: <your_slurm_qos>
mode: SLURM
device: CUDA
ranks_per_node: 8                        # GPUs/node
dataloader_workers: 8
debug: False                             # set True to disable WandB
mem_gb: 0                                # 0 = default partition mem
cpus_per_task: 12
```

---

## Step 6 — Write the top-level training YAML

Copy `uma_sm_direct_pretrain.yaml` to `configs/uma/training_release/uma_sm_solvent_omol.yaml` and change:

| Line | From | To |
|---|---|---|
| `defaults.cluster` | `h100` | `my_slurm` |
| `defaults.dataset` | `uma` | `omol_subset` |
| `defaults.tasks` | `uma_direct` | `omol_only` |
| `dataset_list` | `["oc20", "omol", ...]` | `["omol"]` |
| `train_dataset.dataset_configs` | 5 entries | just `omol: ${dataset.omol_train}` |
| `train_dataset.combined_dataset_config.sampling` | explicit ratios | `{ type: temperature, temperature: 1.0 }` |
| `val_dataset.dataset_configs` | 5 entries | just `omol: ${dataset.omol_val}` |
| `heads` | 5 energy heads + forces | `omol_energy` + `forces` only |
| `job.scheduler.num_nodes` | 16 | start with 1–2 |
| `normalizer_rmsd` | 1.423 | value from Step 2 |
| `run_name` | `uma_sm_direct` | `uma_sm_solvent_omol` |

Also set `element_refs` to point at your computed element-references file (`element_refs/my_omol_refs.yaml` — copy `iso_atom_elem_refs.yaml` and replace values).

The `dataset_list: ["omol"]` change is **critical** — the backbone uses this list to size its dataset embedding (`escn_md.py:269`). A mismatch will either error at instantiation or train a dataset embedding with dead entries.

---

## Step 7 — Hyperparameters worth tuning

The main YAML's top-level keys are the knobs (`uma_sm_direct_pretrain.yaml:31-48` for line refs):

| Knob | Default | When to change |
|---|---|---|
| `max_atoms: 700` | per-batch atom cap (sampler-based, replaces batch size) | shrink to ~200 for short-context smoke tests; grow on bigger nodes |
| `steps: 1680000` | total training steps | scale to dataset size; for ~100k structures start with 100k–200k steps |
| `evaluate_every_n_steps: 10000` (runner) | val cadence | reduce to 1000 for short runs so you see val curves |
| `checkpoint_every_n_steps: 5000` (callback) | ckpt cadence | match eval cadence |
| `lr: 8e-4`, `weight_decay: 1e-3` (optimizer) | AdamW | the UMA defaults are well-tuned; touch only if loss diverges |
| `warmup_epochs: 0.01`, `lr_min_factor: 0.01` (scheduler) | cosine | leave alone unless you change `steps` drastically |
| `clip_grad_norm: 100` | grad clip | leave alone |
| `omol_energy_coef: 30`, `direct_forces_coef: 30` | per-task loss weights | tune if energy/force errors are unbalanced |
| `bf16: True` | mixed precision | set `False` if you see NaNs |
| `solvent_emb_hidden: 16` (K4L2.yaml:43) | solvent MLP width | already set; raise to 32 if the solvent signal seems under-fit |

EMA decay is fixed at `0.999` in `mlip_unit.py:506` and not exposed in the YAML — leave it.

---

## Step 8 — Sanity-check on a single CPU node first (≤ 5 min)

Before burning GPU hours, verify the pipeline end-to-end. Make a copy of your cluster config as `cluster/my_cpu_smoke.yaml` with `mode: LOCAL`, `device: CPU`, `ranks_per_node: 1`, `debug: True`, then:

```bash
fairchem -c configs/uma/training_release/uma_sm_solvent_omol.yaml \
  cluster=my_cpu_smoke \
  steps=10 \
  max_atoms=50 \
  print_every=1
```

This validates: ASE DB loading, solvent name lookup, AtomicData batching, backbone forward, loss masking, optimizer step, and checkpoint write. If `solvent_emb` shapes don't match, this is where you'll see it — not 4 hours into a SLURM job.

---

## Step 9 — Submit the real SLURM run

```bash
fairchem -c configs/uma/training_release/uma_sm_solvent_omol.yaml \
  cluster=my_slurm \
  job.scheduler.num_nodes=2 \
  run_name=uma_sm_solvent_omol_run1
```

`fairchem -c ... cluster=my_slurm` dispatches to `slurm_launch()` (`_cli.py:98-112`) via submitit. Outputs land under `${cluster.run_dir}/${run_name}/`:

```
checkpoints/
├── step_5000/   …
├── step_10000/  …
└── final/
    ├── inference_ckpt.pt   # use this with FAIRChemCalculator
    └── .metadata
```

Resume is automatic — re-submitting the same config picks up the most recent viable checkpoint via `get_most_recent_viable_checkpoint_path` in `components/train/train_runner.py:58-69`.

---

## Critical files to create / modify

| Path | Status |
|---|---|
| `configs/uma/training_release/dataset/omol_subset.yaml` | **new** |
| `configs/uma/training_release/tasks/omol_only.yaml` | **new** |
| `configs/uma/training_release/cluster/my_slurm.yaml` | **new** |
| `configs/uma/training_release/cluster/my_cpu_smoke.yaml` | **new** (smoke test only) |
| `configs/uma/training_release/element_refs/my_omol_refs.yaml` | **new** (output of Step 2) |
| `configs/uma/training_release/uma_sm_solvent_omol.yaml` | **new** (top-level) |
| `configs/uma/training_release/backbone/K4L2.yaml` | already edited |
| `configs/uma/training_release/dataset/{uma,uma_finetune,uma_debug}.yaml` | already edited (kept for completeness; not used by this run) |
| `src/fairchem/core/scripts/create_finetune_dataset.py` | **reuse as-is** for ASE-DB write + normalizer/element-refs |

User-side code (not in the repo): an ORCA / xTB → ASE Atoms parser and a small script invoking `create_finetune_dataset.py`'s helpers.

---

## Verification

1. **Solvent names valid**: in a Python shell, `from fairchem.core.datasets.solvent import list_solvents; assert all(s in list_solvents() for s in your_solvent_set)`.
2. **ASE DB readable**: `AseDBDataset({"src": "/path/to/train"})[0]` returns an Atoms-like dict with `energy`, `forces`, and `solvent` in `info`.
3. **Smoke test (Step 8) completes 10 steps** and writes a checkpoint without OOM, NaN, or shape errors. Confirm the printed loss includes both `omol_energy` and `forces`.
4. **SLURM run logs**: WandB (if `debug: False`) should show energy MAE descending below ~0.1 eV/atom and force MAE descending below ~0.05 eV/Å within the first ~20k steps — coarse sanity, not a publication number.
5. **Inference smoke**: load `checkpoints/final/inference_ckpt.pt` via `pretrained_mlip.get_predict_unit(...)` + `FAIRChemCalculator(predictor, task_name="omol")`, attach to an `ase.Atoms` with `atoms.info["solvent"]` set, call `atoms.get_potential_energy()` — no exceptions and a finite value.
6. **Pre-commit**: run `pre-commit run --files configs/uma/training_release/**/*.yaml` (after installing via `brew install pre-commit`) on every new YAML before committing.
