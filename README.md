# SpaCCES-GAE: reproduction package

Code, trained models and stored results for the article

> A Spatial Coordinate-Conditioned Ego-Subgraph Graph Autoencoder for label-free
> detection, localization and severity ranking of structural damage from strain
> sensor networks (submitted to *Mechanical Systems and Signal Processing*).

SpaCCES-GAE is a dual-view graph autoencoder trained on healthy strain data only.
Its per-sensor reconstruction errors give a run-level detection score with a
label-free threshold, the flagged sensors locate the damage, and a severity index
ranks the damage states.

## Release status

Version 0.1.0 releases the code, the 20 trained models with their training records
(`models/`) and the authors' stored summary results (`results/stored/`). The
processed inputs and per-seed score tables (`data/`) are awaiting the consortium's
approval and will follow in a later release at this same address. Until then the
replays, the rescoring, the retraining and the data tests stop or skip with a
message saying that `data/` is not released yet. With `data/` in place, this
repository lets a reader check every number in the article's results tables,
rescore the 20 trained models and retrain them.

## Quick start (needs `data/`, from the later release)

```bash
pip install -r requirements-replay.txt         # numpy, pandas, scipy, scikit-learn, matplotlib, pytest
python reproduce/replay_all.py --no-rescore     # every table value, from the stored score tables
```

With PyTorch and PyG (`conda env create -f environment.yml`, or `pip install -r requirements.txt`):

```bash
python reproduce/replay_all.py                  # also rescores the 20 released models (~15 min on a laptop CPU)
pytest -q                                       # 14 tests, about 1 minute (without data/, the data tests skip)
python figures/plot_results.py                  # plain redraws of the results figures' data
```

Each replay prints every value it checks next to the value printed in the article
and ends with `ALL CHECKS PASS` or the list of failures.

## Claims and how to check them

| Claim in the article | Command | What it checks |
|---|---|---|
| Detection, localization, severity and complete diagnosis of the 20-seed ensemble (Table 5, Section 4.1, Figs. 7–8, Supplementary Table S-halves-full) | `python reproduce/replay_headline.py` | tau 1.74, AUC 0.975 [0.937, 1.000], 50/50 detected, 0/14 false alarms, 0/7 D6, 50/50 and 11/11 localized, rho 0.962 [0.843, 0.967], rho_min 0.784, CDR 50/50, run 66, per-seed ranges, 12 of 20 seeds complete alone |
| The released models give those numbers | `python reproduce/rescore_models.py` then `replay_headline.py --source rescored` | each model rescored from `data/inputs`; sensor scores within 1e-3 of the stored tables, identical flags and splits, identical Table 5 |
| Comparison with eight graph anomaly detectors, PCA and a dense autoencoder (Table 6) | `python reproduce/replay_comparators.py` | every column of every row, recomputed from each method's per-seed scores; the paired ΔCDR bootstrap intervals and Holm verdicts with the original seeds |
| False alarms under unseen operating conditions (Table 7, Section 5, Fig. 10) | `python reproduce/replay_unseen.py` | thresholds, 6/165 against 62 and 67, all group counts, equal-sensitivity counts, p_Holm 0.003, the run-2 analysis (20/163), the threshold-position medians |
| Ablations (Section 6, Fig. 11, Supplementary Tables S-mechanism and S-rc) | `python reproduce/replay_ablations.py` | all 13 ablation arms, and the Holm p-values 0.003, 0.045, 0.003 and 0.0045 recomputed from the paired bootstraps |
| Training procedure | `python experiments/train_ensemble.py --seeds 0` | retrains a seed from the released inputs (see "Exactness") |

## Layout

```
src/spacces_gae/   model, data loading, scoring chain, localization, metrics, training
reproduce/         one replay script per claim, and replay_all.py
experiments/       retraining from the released inputs
figures/           plain redraws of the data of the results figures
tests/             pytest suite: data invariants, rescoring, replays, hashes, privacy scan
checks/            reconstruction_attack.py (authors only; result in results/privacy/)
tools/             export scripts that derived data/ and models/ (authors only; they read
                   the locations of the restricted files from a private layout file)
data/              released inputs, geometry, score tables (later release)
models/            the 20 trained models and their training records
results/stored/    the authors' original summary files, trimmed to what the replays compare against
results/privacy/   result of the reconstruction check (later release, with data/)
```

## Data

The raw towing-tank strain records belong to the dTHOR consortium (CNR-INM and
Politecnico di Milano) and are not released here. They are available from the
corresponding author on reasonable request. The later release of `data/`
will hold the model's own inputs: for every 4-second window, the per-window min-max normalized variance of
the 60 strain signals, the subgraph context computed from it, and the response
graph. The time series cannot be recovered from these inputs, nor strain in
physical units from the release alone; relative levels between windows are partly
recoverable, and the absolute levels the article itself plots allow an approximate
calibration. `data/README.md` will describe each file, what was withheld and what the
reconstruction tests found.

What needs the restricted records, and therefore cannot be run from this
repository: the feature extraction from raw strain (windowing, trimming, variance),
the construction of the response fingerprints behind S_sim, the measurement panels
of Figs. 1, 2, 4 and 5, and the input-normalization and fingerprint ablations
(N1, N2, R1) of the Supplementary Material.

## Exactness

* The stored score tables were computed on the training machine (Linux, NVIDIA L4).
  Rescoring on another CPU changes sensor scores by at most about 1.4e-4 through
  float32 rounding; every flag, split and reported value is unchanged.
* Training: the released models are the trained originals. Retraining follows the
  same procedure (the first-epoch training loss of seed 0 matches the original to
  six digits on CPU) but not the same floating-point path, so a retrained seed is a
  new draw; compare ensembles, not single seeds. Training was tested on CPU only;
  `--device cuda` is provided but untested in this repository.
* The comparator and ablation values are recomputed from per-seed score tables of
  those models. The comparator detectors themselves (third-party code with the
  adaptations of Supplementary Section S-gnn) are not retrained here.

## Licence and citation

The code is licensed under the MIT licence (`LICENSE`). The released data, models
and stored results (`data/`, `models/`, `results/`) are licensed under CC BY 4.0
(`LICENSE-DATA.md`). If you use them, please cite the article given in
`CITATION.cff`.
