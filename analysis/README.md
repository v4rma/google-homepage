# FA feature analysis (breath focus vs. mind-wandering on Muse-proximal channels)

Decides which spectral features, computed only from channels near the Muse Athena's
electrode sites (AF7/AF8 and TP9/TP10), separate on-task breath focus from
mind-wandering in two open experience-sampling meditation datasets, and exports the
result as `output/fa_feature_spec.json` for the real-time scorer.

Datasets (both CC0, downloaded by the scripts, never committed):

- OpenNeuro [ds001787](https://openneuro.org/datasets/ds001787) (Brandmeyer & Delorme 2018):
  12 expert + 12 novice meditators, 64-ch BioSemi, probes every ~2.5 min asking
  depth of meditation, mind-wandering and tiredness (0-3).
- OSF [3uszv](https://osf.io/3uszv/) (Rodriguez-Larios et al. 2021): 29 meditators +
  29 novices, 19-ch 10-20 + A1/A2, probes every ~1 min asking "breath / thinking about
  something else / distracted by sensation" plus confidence, sleepiness, focus depth.

## Run

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r analysis/requirements.txt
cd analysis/scripts
python download_ds001787.py            # ~6 GB from the OpenNeuro S3 bucket
python download_osf3uszv.py            # ~12 GB of task recordings from OSF
python prepare_ds001787.py             # -> derivatives/ds001787/*.fif + probes.csv
python prepare_osf3uszv.py --workers 3 # -> derivatives/osf3uszv/*.fif + probes.csv
python extract_features.py             # -> derivatives/features.csv
python analyze.py                      # -> output/*.csv, output/fig_*.png, summary.json
python export_spec.py                  # -> output/fa_feature_spec.json
```

Set `FA_DATA_DIR` / `FA_DERIV_DIR` to keep raw data and derivatives elsewhere. Use
`--subjects`, `--max-per-group`, `--files`, `--limit` on the individual scripts for
subsets.

## Pipeline

1. **prepare_*.py** – band-pass 0.5-45 Hz, 50 Hz notch, 256 Hz; keep the Muse-proximal
   channels plus the reference signals (Fpz where recorded, scalp average), and build a
   probe table with labels. ds001787 ratings come from the behavioural logs, aligned to
   the EEG probe triggers (the Status channel never recorded key "0"). The OSF CSVs are
   parsed at 2048 Hz (the only rate at which the 50 Hz line and ~10 Hz alpha peak appear;
   the authors' MATLAB code states 512 Hz).
2. **extract_features.py** – for each probe, window lengths 5/15/30/60 s ending 0.5 s
   before the probe (never overlapping the previous probe's answers), two references
   (`fpz` = Muse-faithful, `avg` = common average as in the papers), two channel
   groups: log band power delta/theta/alpha/beta, relative power, theta/alpha and
   alpha/beta ratios, alpha asymmetry, specparam IAF / alpha peak power / aperiodic
   exponent and offset. Per-channel 2-s segment rejection (150 uV peak-to-peak).
3. **analyze.py** – within-subject paired effects (Cohen's d of mind-wandering minus
   on-task), within-subject stratified CV, leave-one-subject-out with raw vs
   per-subject z-scored features (logistic regression and shrinkage LDA), window-length
   curve, figures.
4. **export_spec.py** – machine-readable spec: bands, Muse channel groups, on-task
   direction per feature and expertise, normalised weights, window, calibration advice,
   expected performance.

Outputs tracked in git: `output/fa_feature_spec.json`, `output/*.csv`, `output/fig_*.png`,
`output/summary.json`.
