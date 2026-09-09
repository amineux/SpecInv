# KerOp filter contract (the only graded joint path)

KerOp writes a **forward** filter contract. SpecInv loads those files and
learns a **spectral filter**. This is the handshake, not a new theorem.

Schema id: `kerop.filter_contract/v1`

SpecInv does **not** mint a parallel `kerop.specinv.spectrum/v1` artifact.
A converter that rewrites KerOp's files into some other schema is not the
graded path.

## What the files are

KerOp's `export-filter-contract` (or `kerop.filter_contract.write_filter_contract`)
writes a stem `filter_contract_v1`:

| File | Role |
|---|---|
| `filter_contract_v1.json` | Header: `schema`, `contract_version`, `producer`, `operators[]` |
| `filter_contract_v1.npz` | Arrays named by `operators[].arrays` (`allow_pickle=False`) |

Load with `json` + `numpy.load`. Do not invent a second schema.

Each `operators[]` record includes:

| Field | Meaning |
|---|---|
| `operator_id` | `kerop.spectral` or `kerop.poisson` (KerOp may rename the latter) |
| `n` | KerOp **training sample size** (default 150), not the eigenvalue count |
| `n_features` / `feature_count` | Random-feature count `M` |
| `arrays.eigenvalues` | NPZ key for the descending forward spectrum |
| `arrays.rf_spectrum` | NPZ key for the empirical RF spectrum |
| `reproduce` | CLI / train sizes / lambda grid / `task_kwargs` that reproduce the bar |
| `recorded_bar` | Copied medians from KerOp's committed wall-time JSONs |

`kerop.poisson` is KerOp's **1-D Dirichlet** wall-time operator (12
collocation points). It is **not** FEM Poisson. It does not have enough
eigenvalues for SpecInv 11/11. SpecInv accepts documented aliases
(`kerop.poisson_1d`, `kerop.dirichlet_poisson_1d`, `kerop.walltime_1d`,
`kerop.walltime_poisson_1d`) if KerOp renames it.

## Joint instance

* Graded SpecInv target: **`kerop.spectral`** — product-set eigenvalues of
  KerOp's `SpectralOperatorModel(r=0.5, b=0.5, n_modes=512, output_dim=6)`.
  The exported 1-D array has 3072 values; SpecInv uses the leading 2048 as
  singular values. Ill-posedness is KerOp's `p = 1/b = 2`. SpecInv's Sobolev
  prior stays `s = 1.5`. Theory rate is `s/(s+p)`, not a hard-coded 0.5.
* KerOp matched-risk bar: live `run_walltime_benchmark` using that operator's
  `reproduce` block. Print the **live** median. KerOp's published
  spectral-quick median (~27.7×) is context, not this run's claim.
* `fixtures/filter_contract_v1.{json,npz}` is a **copy of KerOp's export**,
  not a SpecInv-minted table.

## How SpecInv finds the files

`specinv-joint` resolves, in order:

1. `--contract PATH` (alias: `--spectrum`)
2. `--kerop-root` / `KEROP_ROOT` / a sibling KerOp checkout that already
   contains `filter_contract_v1.{json,npz}`
3. this repo's committed fixture (copied from KerOp)
4. `--write-contract`: ask **KerOp** to write its own files

If the files are missing or `schema` is not `kerop.filter_contract/v1`, the
script **fails**. It does not rerun SpecInv's built-in 1-D operator and call
that a joint result.

## What this is not

* Not a new rate theorem.
* Not a 100× (or any other invented-factor) claim.
* Not a re-claim of KerOp's ~28× if the live median is lower.
* Not GPU RF, imaging, FEM, NeuroFEM, a localizer, or Loihi.
