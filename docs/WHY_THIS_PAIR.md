# Why KerOp + SpecInv together — and why that still isn’t a breakthrough

## Why they’re a credible pair

Both repos attack the **same scientific niche from opposite ends**:

| | **KerOp** | **SpecInv** |
|---|---|---|
| Paper | [arXiv:2603.00971](https://arxiv.org/abs/2603.00971) (AISTATS 2026) | [arXiv:2603.20602](https://arxiv.org/abs/2603.20602) |
| Object | Learn a **forward** operator with operator-valued kernels / random features | Learn a **regularised inverse** via an adaptive **spectral filter** |
| Math spine | Spectral filtering of OVK / RF (Tikhonov, Landweber, rates) | Spectral filtering of singular coefficients (SC-Net vs Tikhonov/TSVD) |
| Gap we filled | No public paper code | No matching public impl |
| Falsifiable bar | Rate order ≈ Thm 3.4; RF beats exact KRR on **wall-clock** at matched risk | Rate ≈ δ^{1/2}; zero-shot N=256→2048; beat Oracle Tikhonov |

Together they say: *spectral filtering is the right language for both operator learning and inverse problems, and we shipped the missing reference software with numbers, not vibes.*

That is a coherent **portfolio**: one forward SciML library, one inverse SciML library, same era of theory, same honesty standard (document where the paper is ambiguous).

## Why it is *not* as impressive as the hype suggests

1. **Reference code ≠ discovery.** The theorems already exist. A clean reimplementation is valuable engineering — not a new Nature/NMI result.

2. **SpecInv’s flashiest paper claim did not fully reproduce.** Under one consistent noise model, SC-Net and Oracle Tikhonov share ~the same rate exponent; the advertised 0.50 vs 0.42 gap looks like a **noise-convention mismatch**. Our win is a **constant-factor** improvement and interpretability, not a new asymptotic regime.

3. **KerOp’s RF win is speed, not accuracy.** At matched sample size, exact operator-valued KRR is still more accurate. Random features matter when the Gram matrix is too big — a systems win.

4. **Laptop-scale synthetics.** 1D Fredholm / spectral operators verify theory. They are not turbulent CFD or clinical inverse problems.

5. **No hardware / no field deployment.** No Loihi, no lab instrument, no production digital twin.

6. **Repro debt remains.** Heavier Darcy tables, elliptic source problems, and 2D–3D are still out of scope. Re-run the scripts before citing numbers.

## Simulation folders

- `KerOp/results/` — wall-time frontiers, rate / feature-threshold outputs
- `SpecInv/results/` — `summary.json` PASS/FAIL, convergence / zero-shot / filters

Cite the **papers first**, then this software as independent verification.

## The one joint path (not a new result)

Team verdict: keep **one** path. KerOp writes `kerop.filter_contract/v1`
(`filter_contract_v1.{json,npz}`; see [KEROP_SPECTRUM.md](KEROP_SPECTRUM.md)).
SpecInv loads **those** files — not a SpecInv-minted parallel schema — and
learns the inverse filter. `specinv-joint` runs both bars on that path:

* KerOp: live matched-excess-risk wall-clock comparison using the contract's
  `reproduce` settings. Print the live median. The published spectral-quick
  median is ~27.7×; do not re-claim that figure if this run is lower.
* SpecInv: the existing 11/11 suite (rate vs `s/(s+p)` for **this** spectrum,
  zero-shot, beat Oracle Tikhonov, filter checks) on a target sampled from
  KerOp's `kerop.spectral` eigenvalues.

If the KerOp contract files are missing or the schema id mismatches, the
joint script fails. It will not rerun the old 1-D suite and call that a
joint result. `kerop.dirichlet1d` is 1-D Dirichlet wall-time, not FEM.
This is plumbing, not a theorem. The live graded run on this path is
**JOINT FAIL** (SpecInv 8/11, KerOp live ~7×) — wiring, not a joint win.
