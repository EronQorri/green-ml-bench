# Chapter 2 Review — Theoretical Background

Review of `thesis_latex/chapters/02_theoretical_background.tex` (367 lines).
Line numbers refer to that file.

## Overall take

This is a strong, comprehensive theory chapter with an unusually consistent Green AI
through-line: nearly every concept is tied back to the ecological-cost argument rather
than presented in isolation, which is exactly what a thesis of this framing needs. The
math is well formalized, the citations are dense and appropriate, and several passages
show genuine sophistication (the histogram-vs-GPU efficiency distinction, the CPU/GPU
suitability reasoning, the honest caveats on OOB and GPU random forests). The CO2eq
terminology is also far more disciplined here than in chapter 4.

The main weaknesses are a handful of small factual/typographic errors, a few overloaded
math symbols, and one section that drifts toward related-work territory. Nothing
structural is broken.

---

## Must-fix errors

### 1. Duplicate citation
Line 172: `... directly in device memory \parencite{...2017} \parencite{...2017}.` — the
same `mitchellAcceleratingXGBoostAlgorithm2017` key is cited twice back-to-back. Delete
one.

### 2. Typo in a paragraph heading
Line 184: `\paragraph{Activations Functions.}` should be `Activation Functions`.

### 3. "Negative Binomial Log-Likelihood" is the wrong distribution name
Line 102: binary cross-entropy is the negative log-likelihood of a **Bernoulli** trial
(equivalently the binomial deviance with n=1). "Negative Binomial" names a *different*
distribution entirely (the negative binomial / count distribution). A statistics-literate
reviewer will read this as an error. Use "negative Bernoulli log-likelihood" or, matching
Hastie's wording, "binomial deviance".

### 4. `E = W \cdot s` uses units as variables
Line 31: the energy formula is written `$E = W \cdot s$`, mixing the unit symbols W (watts)
and s (seconds) into what should be a relation between *quantities*. The correct form is
`$E = P \cdot t$` (power times time). The line already defines power and time in words
right before it, so the quantities are available. Keep the "(watt-seconds = joules)" unit
note separately if you want it.

---

## Consistency issues that touch correctness

### 5. The symbol `M` is overloaded
- `M` = number of classes (line 98, line 220 `$M \times M$`, lines 248-253).
- `M` = number of trees in the forest (line 146: "multiplied across all `$M$` trees").

Same symbol, two unrelated meanings, in the same chapter. Random forests conventionally
use `B` (Breiman), `T`, or `n_trees`. Rename the tree count and reserve `M` for classes.

### 6. Learning-curve power law contradicts itself
Line 203: "the rate of improvement eventually diminishes, typically following an
**inverse** power law `$P(n) \propto n^\alpha$` with `$\alpha \in (0,1)$`." Two problems:
(a) "inverse" implies a negative exponent, but `$n^\alpha$` with positive `$\alpha$` is
not inverse; (b) `$P(n) \propto n^\alpha$` grows without bound, so it never plateaus,
which contradicts the "diminishing returns / asymptote" claim the paragraph is making.
The standard formulation states it for the *error*: `error $\propto n^{-\alpha}$` (or an
asymptotic form `$a\,n^{-\alpha} + c$`). Either switch to error, or drop "inverse" and
acknowledge the missing asymptote.

### 7. Carbon-intensity symbol switches mid-sentence
Line 36: "multiplying the total energy consumed (`$E$`) by \gls{ci}, yielding
`$C = E \cdot I_C$`." The prose uses `\gls{ci}` and the formula uses `$I_C$` in the same
breath. Pick one symbol for the formulas (`$I_C$` is fine) and introduce it explicitly the
first time, then use it consistently (line 265 again refers to it as `\gls{ci}`).

### 8. Reused Greek symbols across model sections
`$\gamma$` is the XGBoost tree-size penalty (line 164) and also the BatchNorm scale
parameter (line 188); `$\lambda$`, `$\mathcal{L}$`, and `$T$` are likewise reused across
subsections. This is common in the literature and each use is locally unambiguous, so it
is not a hard error, but if you want maximum cleanliness, a one-line "notation is local to
each subsection" note, or disambiguating the BatchNorm params, would remove any doubt.

---

## Theoretical / methodological points worth addressing

### 9. The Software Measurement subsection reads like related work
Lines 77-88 (web calculators -> packages -> CodeCarbon lineage) is a tool-history
narrative more than theoretical background, and it likely overlaps with chapter 3. Either
trim it to the minimum needed to justify the CodeCarbon choice and push the historical
detail to related work, or explicitly frame it as "tooling background" so the boundary
with chapter 3 is intentional.

### 10. "Internal covariate shift" is a contested explanation
Line 188 presents ICS as *the* reason BatchNorm works, following Ioffe & Szegedy.
Subsequent work (Santurkar et al., 2018) showed BN does not actually reduce ICS and
attributed its benefit to a smoother loss landscape. Citing the original is fine, but a
half-sentence caveat ("though the precise mechanism is debated") would protect you from a
reviewer who knows the follow-up.

### 11. Uneven depth across the model subsections
The MLP gets five run-in sub-paragraphs (backprop, activations, batchnorm, dropout, early
stopping) while logistic regression gets three short ones. This is partly justified — the
MLP genuinely has more moving parts — but the imbalance is large. Consider whether every
MLP regularizer needs full treatment in a theory chapter, or whether some (dropout, early
stopping) can be compressed, since not all are central to the energy-efficiency argument.

### 12. The "3.4 months doubling" stat is cited via a secondary source
Line 11 attributes the compute-doubling figure to `verdecchiaDataCentricGreenAI2022a`, but
it originates in OpenAI's "AI and Compute" analysis. Citing the secondary source is
acceptable; just be aware a reviewer may flag it, and consider an "as reported by" framing.

---

## Style and cleanup

- **Trailing space inside `\gls`.** `Green \gls{ai} ` with a trailing space before the
  closing brace appears repeatedly (lines 13, 58, 203, 209, and others), and line 60 has a
  double space after `\textit{\gls{simt}}`. Cosmetic, but a sweep is easy.
- **`\citeauthor{x} \parencite{x}` pattern.** Lines 11, 172 render as "Author et al.
  (2020) [ref]", which looks doubled. If intentional (named attribution + bracketed cite),
  keep it but apply it consistently; otherwise collapse to `\textcite`.
- **CO2 subscript usage.** Most raw `CO\textsubscript{2}` here are legitimate product
  names (`ML CO\textsubscript{2} Impact`, lines 17, 80, 84) so they are fine. The one to
  reconsider is line 88, `kgCO\textsubscript{2}eq`, which is used as a unit literal — decide
  whether it should be `\gls{co2eq}`-styled or left as a verbatim unit, and apply the same
  choice anywhere the storage unit is named. (Overall this chapter is far cleaner than
  chapter 4 on this front.)
- **Glossary commands inside a math fraction.** Line 34 puts `\gls{gco2eq}` and `\gls{kwh}`
  inside `$\frac{...}{...}$`. Verify this renders cleanly (glossary hyperlinks in math mode
  can misbehave); a plain-text unit in math may be safer.
- **Section placement of Cross-Validation.** It sits as a standalone top-level `\section`
  (line 275) between Evaluation Metrics and HPO, yet it is conceptually a sub-topic of the
  estimation/HPO machinery. The current order works as a bridge (the Selection Bias
  paragraph at line 295 hands off neatly into HPO), but consider whether CV reads better as
  a subsection under HPO or grouped with metrics.

---

## What works well

- **The histogram-vs-GPU efficiency distinction (line 172)** is the most sophisticated
  point in the chapter: separating the genuine algorithmic work-reduction (binning) from
  the mere time-to-power redistribution of GPU acceleration, and tying it explicitly to
  Schwartz et al.'s definition of efficiency. This is exactly the kind of reasoning the
  thesis is built on.
- **CPU vs GPU suitability (lines 55-64)** is well argued and honestly caveated — the
  small-dataset GPU-overhead point (line 64) pre-empts an obvious objection.
- **Honest caveats throughout:** OOB scoring is optional and flagged as such (line 142),
  the GPU-RF footnote (line 142) is candid about implementation dependence, and the
  tree-vs-DL-on-tabular trade-off (line 197) is well sourced.
- **Consistent Green AI framing.** Almost every subsection closes by relating the concept
  back to ecological cost (learning curves, HPO trial count, composite-metric limits),
  giving the chapter a coherent spine rather than a disconnected concept list.
- **The HPO progression** (GS -> RS -> BO -> TPE -> Optuna, with the complexity table at
  line 339) is logically ordered and each step is motivated by the previous one's
  limitation.
- **The Selection Bias paragraph (line 295)** correctly sets up the optimistic-bias problem
  that chapter 4's held-out-test design resolves — the theory and methodology are aligned.

---

## Suggested fix order

1. **Unambiguous fixes (safe to apply directly):** items 1, 2, 3, 4, plus the trailing-space
   sweep and the math-mode glossary check.
2. **Judgment calls (decide first):** items 5-12, and the Cross-Validation placement.
