# Chapter 4 Review — Methodology and Implementation

Review of `thesis_latex/chapters/04_methodology_and_implementation.tex` (741 lines).
Line numbers refer to that file.

## Overall take

Strong methodology chapter, unusually thorough for a bachelor thesis: well-reasoned
measurement boundaries, an original and honestly caveated CPU power-correction story,
and a mostly coherent experiment-to-research-question framing. The validation strategy
reads cleanly now that the held-out split is in place. The main problems are consistency
issues (some touching correctness) and a few leftover artifacts from the chapter 4/5 merge.

---

## Must-fix errors

### 1. The cross-validation class is named three different ways, two of them wrong
- Line 38: `\path{Stratified_Shuffle_KFold}`
- Line 390: `\texttt{StratifiedKFold}` (correct)
- Line 432: `\texttt{ShuffleStratifiedKFold}`

Only line 390 names a real scikit-learn class. There is no `Stratified_Shuffle_KFold`
or `ShuffleStratifiedKFold`. You are using `StratifiedKFold(shuffle=True)`. Both an
inconsistency and a factual error a technical reviewer will catch immediately.
**Standardize all three to `StratifiedKFold`.**

### 2. Leftover TODO / German note in the text
Line 36: `The full implementation is \todo{noch private} publicly available`.
Resolve before submission and reword once public vs. private is decided.

### 3. Typo in a paragraph heading
Line 676: `\paragraph{Dateset.}` should be `Dataset`.

### 4. `CO\textsubscript{2}` vs `\gls{co2eq}` is inconsistent across the chapter
Own convention (and saved preference) is `\gls{co2eq}`. The Metrics section uses it
correctly (lines 330, 335, 632), but the entire **Emissions Measurement** section and
the results-schema tables fall back to raw `CO\textsubscript{2}` (lines 447, 450, 453,
504, 512, 515, 517, 576, 577, 587, 606, 689, 692, 695, and more). Most widespread
inconsistency in the chapter. Pick `\gls{co2eq}` and sweep.

---

## Consistency issues that touch correctness

### 5. The two MLP parameter-count tables use different counting conventions
- `tab:mlp_architecture` (tuned, line 226) **includes** BatchNorm parameters.
  Verified Wine: 1,024 + 17,152 + 771 = 18,947 only if BN scale/shift params are added.
- `tab:mlp_architectures` (variation grid, line 660) **excludes** BatchNorm.
  Width-128/1-layer = 3,970 and width-512/4-layer = 803,842 match exactly *only without* BN.

Two tables in the same chapter present "parameter count" under opposite conventions,
even though line 228 says every hidden block has BatchNorm. Either recompute the
variation table with BN, or state that the variation counts are linear-layer-only and
explain why.

### 6. "Only the resulting emissions are reported" is contradicted by the rest of the chapter
Line 322 states energy is not reported, only emissions. But the results schema reports
four energy columns (lines 578-581), Exp. 4 derives energy back from emissions
(eq. on line 695), and CF1 plus the lifecycle analysis both lean on energy
(`energy_per_inference_wh`). Soften line 322 to something like: "emissions are the
primary reported metric, with component energy retained for the scaling and lifecycle
analyses."

### 7. The carbon-intensity factor used in `compute_corrected_co2` is never stated, yet Exp. 4 and Exp. 5 depend on it
Exp. 4 (lines 692-695) recovers energy by inverting the static 381 g/kWh factor from the
*recorded* CO2 value, but the recorded `co2eq_kg` is the *corrected* figure
(hardware CPU + CC GPU/RAM). For that inversion to be valid, the corrected value must
also have been built with 381 g/kWh throughout. Meanwhile Exp. 5 (line 546) *derives*
intensity per run as `I_C = C_train / E_train`. If everything uses 381, that derivation
is just 381 every time and is redundant; if it isn't, then Exp. 4's inversion is wrong.
Verify the factor is handled identically in all three places, state it explicitly where
`compute_corrected_co2` is described (lines 511-512), and specify which column Exp. 4
inverts (`co2eq_kg` vs `co2eq_codecarbon_kg`) on line 692.

### 8. XGBoost tuning: "tuned once per dataset, device-independent" vs. the `tune_xgb_cpu_wine` example name
Line 184 argues CPU and GPU converge to the same config, so XGB is tuned once. The
eight-tuning-run count (2 RF + 3 XGB + 3 MLP) on line 509 only works if XGB is tuned
once per dataset. But the same line gives `tune_xgb_cpu_wine` as an example project name,
implying a CPU-specific tuning run. Pick one story. If tuning is device-independent, the
project name should be `tune_xgb_wine`.

---

## Methodological points worth addressing

### 9. The class-imbalance claim is vague and possibly inaccurate
Line 350: "two of the three datasets exhibit class imbalance." HIGGS is roughly 53/47
(essentially balanced) and Wine is moderately balanced across three classes. The clearly
imbalanced one is Credit (~78/22). Name the datasets and ideally give the ratios,
otherwise the justification for stratification and weighted F1 rests on a claim a
reviewer could dispute for HIGGS.

### 10. Experiment-to-RQ mapping is not 1:1, and Exp. 4 maps to no RQ
From line 625: RQ1 -> Exp1, RQ2 -> Exp3, RQ4 -> Exp2, RQ5 -> Exp5, and RQ3 is folded
into Exp2 ("feeds into RQ3", line 652). That leaves Exp. 4 (carbon intensity) attached to
no research question. May be intentional (it validates the measurement rather than
answering an RQ), but state that explicitly, otherwise "five RQs, five experiments"
invites the reader to look for a mapping that isn't there.

### 11. "Exact reproducibility" is overstated for the GPU MLP
Line 290: `torch.manual_seed` alone does not make cuDNN deterministic; deterministic
flags are needed. Consider "reproducible up to GPU non-determinism" or note it as a
known caveat.

### 12. "Converges to the same optimal configuration regardless of hardware" is a strong claim
Line 184: floating-point differences between CPU and GPU histogram builds can produce
slightly different splits. The claim is fine for justifying hyperparameter reuse, but
soften to "the optimal configuration is not expected to depend on the execution device."

### 13. Per-inference energy includes Python loop overhead
Line 537: for sub-microsecond predictors (LR), most of the measured
`energy_per_inference_wh` over the 30 s window is the `while` loop, `perf_counter` calls,
and pipeline dispatch, not model compute. The single-row design is well justified
(lines 527-528), but a sentence acknowledging that per-inference energy for the cheapest
models is dominated by harness overhead would strengthen the validity argument.

---

## Style and cleanup

- **Leftover merge labels.** Duplicate `\label`s (lines 2-3: `experimental_design` +
  `implementation`; lines 48-50: three labels on one section). Prune to one canonical
  label each and update cross-refs.
- **`\Cref` vs `\ref`.** Line 52 uses `Table~\ref{tab:datasets}` while the rest uses
  `\Cref`. Line 732 mixes `\eqref` and `\Cref` for equations. Standardize.
- **Thousands separators in prose.** Line 52 writes `30,000` and `11,000,000` with plain
  commas, while tables use `{,}`.
- **Stray commas.** Line 689 "for Germany which is the default" needs a comma; line 692
  "from the year 2024, is fetched" has an extra one.
- **Grammar, line 38.** "It provided the `LogisticRegression` and `RandomForestClassifier`
  implementations, also the ..." reads awkwardly. Use "as well as".
- **Tense mixing.** Alternates past tense for experiments ("were run") and present for
  code behavior ("stores, reads, returns"). Defensible split but not applied
  consistently. Worth one deliberate pass.
- **Altitude of some implementation detail.** Passages like the `module__*` prefix
  (line 290), `csv.DictWriter` writing a header on first creation (line 560), and joblib
  loading (line 531) are quite low-level for a methodology chapter. A reviewer focused on
  method might suggest moving the most granular bits to an appendix.

---

## What works well

- Measurement-boundary reasoning (lines 339-340, 452-453) is clear and well-motivated:
  excluding data loading so emissions reflect algorithmic cost is the right call and it
  is defended.
- The CPU power-correction section is the most original part. The concrete 6.6 W vs
  57.8 W example (line 512) and the honest Windows/RAPL/admin limitations (lines 514-517)
  are exactly right.
- The Wine small-test-set limitation (lines 358-359) is handled with integrity rather
  than buried.
- The scaling experiment's fixed held-out test pool across seeds and sizes (line 646) is
  methodologically sound, and the over-specification-is-conservative argument (line 649)
  is genuinely good reasoning.
- The CF1 metric is well-motivated, the unit choice is justified with actual ranges, and
  the "absolute, not normalization-based" property (line 335) is a real advantage you
  correctly highlight.

---

## Suggested fix order

1. **Unambiguous fixes (safe to apply directly):** items 1, 2, 3, 4, plus the stray
   commas, duplicate labels, and `\Cref`/`\ref` standardization.
2. **Judgment calls (decide first):** items 5-13.
