# fractal-threshold

![alpha1 during a real exercise stress test](docs/figures/alpha1_stress_test_306.png)

<sub>MIT-BIH ST Change Database record 306, an exercise stress test with reference beat annotations (PhysioNet, ODC-BY). α1 in 120 s windows every 5 s; HRVT1 and HRVT2 from a linear α1~HR fit on the ascending limb only.</sub>

DFA-α1 from RR intervals. It estimates the aerobic (α1 = 0.75) and anaerobic (α1 = 0.50) HRV thresholds from a chest-strap or ECG recording, and computes the usual recovery metrics (RMSSD, SDNN, pNN50, Lomb–Scargle LF/HF, Poincaré SD1/SD2). The thing I did not expect: on the same ECG at the same HR of 124 bpm, swapping the R-peak detector moves α1 from 1.00 to 0.61, because a few ms of timing jitter is enough to fake a threshold at exercise HR.

<!-- TODO(Tommaso): dictate 3-4 lines in your own words -->

## Install and use

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[fit]"          # add [validate] to rerun the PhysioNet checks

hrv-threshold alpha1 ride.fit --thresholds --plot alpha1.png --csv alpha1.csv
hrv-threshold metrics morning_h10.txt --json --plot poincare.png
hrv-threshold alpha1 examples/synthetic_ramp.txt --thresholds   # bundled synthetic demo
```

```python
from hrvthreshold import read_rr, alpha1_timeseries, estimate_thresholds, all_metrics
rr = read_rr("ride.fit")                       # ms
s = alpha1_timeseries(rr)                      # 120 s / 5 s windows
v = s.valid(max_artifacts=0.05)
th = estimate_thresholds(s.alpha1[v], s.hr[v])
print(th.hrvt1, th.hrvt2, th.r2, th.reliable)
```

`.fit` files are read with `fitdecode` from the `hrv` message (up to five RR values per message, `65.535 s` means invalid), so turn on HRV/RR logging on the watch and pair a chest strap. CSV/TXT can hold one RR per line (ms or s, auto-detected) or a table with an `RR`/`IBI` column, such as Polar Sensor Logger's `Phone timestamp;RR-interval [ms]`.

Tests and the PhysioNet reproduction:

```bash
pip install -e ".[dev]" && pytest -q        # 41 tests, ~2 s, no network
pip install -e ".[validate]"
python scripts/validate_physionet.py         # downloads ~55 MB from PhysioNet
python scripts/make_figures.py               # regenerates docs/figures
```

No personal health data is in this repository. Put your own recordings in `private/`, which is git-ignored, as is `*.fit`.

## Results

On synthetic signals (N = 4096, 50 reps) α1 over n = 4–16 gives 0.59 ± 0.02 for white noise (theory 0.50), 1.02 ± 0.02 for pink 1/f and 1.50 ± 0.02 for Brownian. The +0.09 on white noise is the known small-scale bias of DFA-1 (Kantelhardt et al., 2001), and the 0.75/0.50 cut-offs were calibrated with the same estimator in Kubios. Against [`nolds`](https://github.com/CSchoel/nolds) `dfa` configured identically, |Δα| < 1e-9 on 15 synthetic configurations and max |Δα| = 8.2e-15 on every valid window of all 38 real PhysioNet records. Raw numbers are in [`results/`](results/).

On real annotated ECG:

| dataset | n | result |
|---|---|---|
| [Fantasia](https://physionet.org/content/fantasia/1.0.0/), 2 h supine rest | 10 | α1 = 1.05 ± 0.17 (range 0.73–1.27) |
| [MIT-BIH ST Change DB](https://physionet.org/content/stdb/1.0.0/), stress tests reaching ≥ 140 bpm | 12 | α1 1.25 at lowest-HR decile, 0.59 at highest; drop > 0.2 in 9/12; median Spearman ρ(α1, HR) = −0.53 |
| same, reliable threshold fits | 7 / 12 | median HRVT1 = 89 % of peak HR, HRVT2 = 95 % |

These are clinical stress-test patients with no gas-exchange reference, so this shows α1 behaves as expected on real data. It says nothing about threshold accuracy against VT1/VT2. Three records reach high HR with no α1 decline (309, 310, 320) and the reliability gate rejects their fits.

On the [Wrist PPG During Exercise](https://physionet.org/content/wrist/1.0.0/) chest-ECG lead (no beat labels) I compared `wfdb` XQRS with a band-pass (8–30 Hz) detector with parabolic sub-sample refinement. The band-pass detector gave a valid window on 15/19 records, XQRS on 11/19, and on 3 records their median α1 differs by more than 0.15. The reason is that a timing error $e_k$ on beat times adds $e_{k+1} - e_k$ to every RR interval, which is anti-correlated noise. At 150 bpm with an RR SD of 4 ms, jitter of σ = 2 ms takes a true α = 1.0 down to α1 ≈ 0.74. At rest (RR SD 40 ms) the same jitter does nothing. One sample at 256 Hz is already 3.9 ms.

## How it works

DFA (Peng et al., 1995) integrates the RR series $x_1,\dots,x_N$ into a profile

$$
y(k) = \sum_{i=1}^{k} \left( x_i - \bar x \right),
$$

splits it into $N_n = \lfloor N/n \rfloor$ boxes of $n$ beats, removes a least-squares line $\hat y_\nu$ from each box, and takes the RMS of what is left:

$$
F(n) = \sqrt{ \frac{1}{N_n\, n} \sum_{\nu=1}^{N_n} \sum_{k \in \nu} \bigl( y(k) - \hat y_\nu(k) \bigr)^2 }.
$$

For a self-similar signal $F(n) \propto n^{\alpha}$, and α1 is the OLS slope of $\log F$ against $\log n$ over $n \in \{4, \dots, 16\}$ beats, the Kubios convention used by the threshold papers. 0.5 is white noise, 1.0 is $1/f$ (healthy resting heart), 1.5 is Brownian; for a $1/f^{\beta}$ spectrum $\alpha = (\beta+1)/2$. With rising intensity, vagal withdrawal breaks the correlations and α1 falls toward 0.5 and below. Rogers et al. (2021) matched α1 = 0.75 to the first ventilatory threshold and α1 = 0.50 to the second.

Windows follow Rogers et al. (2021, *Front. Physiol.*): 120 s, recomputed every 5 s, so a window holds about 120 beats at 60 bpm and about 340 at 170 bpm. A single window has an SD of 0.05–0.09, so thresholds come from an OLS fit $\alpha_1 = a\,\mathrm{HR} + b$ over all windows with $0.25 \le \alpha_1 \le 1.05$:

$$
\mathrm{HRVT1} = \frac{0.75 - b}{a}, \qquad \mathrm{HRVT2} = \frac{0.50 - b}{a}
$$

A fit is `reliable` only if $R^2 \ge 0.5$ and neither threshold is extrapolated outside the observed HR range. Before any of this, beats outside the physiological range or more than 20 % off an 11-beat local median are replaced by cubic spline, and windows with more than 5 % corrected beats are dropped.

Recovery metrics follow the Task Force (1996). $\mathrm{SD1}^2 = \tfrac12 \mathrm{Var}(\Delta x)$, $\mathrm{SD1}^2 + \mathrm{SD2}^2 = 2\,\mathrm{Var}(x)$. LF (0.04–0.15 Hz) and HF (0.15–0.40 Hz) come from a Lomb–Scargle periodogram on the unevenly sampled tachogram with no resampling, Parseval-scaled so that $\int P\,df$ equals the variance of the detrended series (absolute powers in ms²).

## Limitations

- Chest strap or ECG only. Optical PPG has far more timing jitter, which biases α1 low exactly where thresholds live.
- Artifact correction is a simple rule, not the Lipponen & Tarvainen (2019) algorithm behind Kubios "automatic". Expect small differences from Kubios on dirty data.
- Thresholds need an incremental test that crosses 0.75 (and 0.5); otherwise the fit extrapolates and is flagged unreliable.
- The DFA estimator is verified exactly. The threshold mapping is not, because no open dataset here has gas exchange. Simultaneous RR + VO₂ data would be welcome.
- NN intervals only. Arrhythmias, pacing or atrial fibrillation make α1 meaningless.
- Not a medical device.

## References

- Peng C-K, Havlin S, Stanley HE, Goldberger AL. Quantification of scaling exponents and crossover phenomena in nonstationary heartbeat time series. *Chaos* 5:82 (1995). doi:10.1063/1.166141
- Kantelhardt JW et al. Detecting long-range correlations with detrended fluctuation analysis. *Physica A* 295:441 (2001). doi:10.1016/S0378-4371(01)00144-3
- Gronwald T, Rogers B, Hoos O. Fractal correlation properties of heart rate variability: a new biomarker for intensity distribution in endurance exercise and training prescription? *Front. Physiol.* 11:550572 (2020). doi:10.3389/fphys.2020.550572
- Rogers B, Giles D, Draper N, Hoos O, Gronwald T. A new detection method defining the aerobic threshold for endurance exercise and training prescription based on fractal correlation properties of heart rate variability. *Front. Physiol.* 11:596567 (2021). doi:10.3389/fphys.2020.596567
- Rogers B, Giles D, Draper N, Mourot L, Gronwald T. Detection of the anaerobic threshold in endurance sports: validation of a new method using correlation properties of heart rate variability. *J. Funct. Morphol. Kinesiol.* 6:38 (2021). doi:10.3390/jfmk6020038
- Rogers B, Giles D, Draper N, Mourot L, Gronwald T. Influence of artefact correction and recording device type on the practical application of a non-linear heart rate variability biomarker for aerobic threshold determination. *Sensors* 21:821 (2021). doi:10.3390/s21030821
- Lipponen JA, Tarvainen MP. A robust algorithm for heart rate variability time series artefact correction using novel beat classification. *J. Med. Eng. Technol.* 43:173 (2019). doi:10.1080/03091902.2019.1640306
- Task Force of the ESC and NASPE. Heart rate variability: standards of measurement, physiological interpretation and clinical use. *Circulation* 93:1043 (1996). doi:10.1161/01.CIR.93.5.1043
- Data: Iyengar N et al., *Am. J. Physiol.* 271:R1078 (1996), doi:10.1152/ajpregu.1996.271.4.R1078 (Fantasia, doi:10.13026/C2RG61). Albrecht P., MIT M.S. thesis (1983) (ST Change DB, doi:10.13026/C2ZW2H). Jarchi D, Casson AJ, *Data* 2(1):1, doi:10.3390/data2010001 (Wrist PPG, doi:10.13026/C2PQ1X). PhysioNet: Goldberger AL et al., *Circulation* 101:e215 (2000).

MIT © 2026 Tommaso Maggiolo. See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff).
