# Dossier composition analysis

**What this covers:** occurrence, co-occurrence, dispersion, and order of the 10 tracked document types across the full corpus of 3,343 dossiers (114,986 predicted pages / 80,549 predicted document instances), predicted by the `vision_efficient` classifier.

**Status:** this run uses the current classifier. A better-performing model's predictions are expected; the entire pipeline reruns against new predictions with no code changes (`make -C scripts/dossier_composition all PREDICTIONS=... TEST_PREDICTIONS=...`) — treat the numbers below as the current best answer, not final.

## Why this needed a correction layer, not just counting

The classifier's per-type reliability is very uneven — from near-perfect (D.1: 96.5% recall) down to poor (Testimonial medical form: 54.8%, Testimonial labour: 61.2%), with precision problems concentrated elsewhere (D.2's raw prevalence turned out to be substantially inflated by misclassified "Other" pages). Naive counts on predicted labels would be systematically wrong, and wrong by different amounts for different types — not a uniform noise level you can shrug off.

**Shared method across all four analyses:** a Bayesian confusion matrix (hierarchical Dirichlet-multinomial, partially pooled across types) fit from a 307-page held-out test set, used to multiply-impute each predicted document instance's plausible "true" type (200 imputations, each also resampling the confusion matrix's own posterior). Every statistic below is computed on both the raw predicted labels ("naive") and pooled across the 200 imputations ("corrected"), so the gap between them shows exactly how much the classifier's unevenness would have distorted a naive reading. Full detail: [confusion_matrix.py](../../scripts/dossier_composition/confusion_matrix.py), [imputation.py](../../scripts/dossier_composition/imputation.py).

Recall per tracked type (posterior mean, from the confusion matrix):

| type | recall | type | recall |
|---|---|---|---|
| D.1 | 0.965 | Report of selection and medical officers | 0.890 |
| D.2 | 0.945 | Registration card | 0.767 |
| NAMA agreement | 0.932 | Approval notice | 0.796 |
| DM.1 | 0.926 | Testimonial labour | 0.612 |
| Judicial and political background check | 0.914 | Testimonial medical form | 0.548 |

D.2's *precision* (not shown in this table) is the real outlier — its raw prevalence looked far higher than corrected, because many misclassified "Other" pages land on D.2 specifically (see Occurrence, below).

## 1. Occurrence

*[occurrence_naive_vs_corrected.png](occurrence_naive_vs_corrected.png)*

D.1 is near-universal (94.7% of dossiers, [93.5%, 95.4%]) — the closest thing to a mandatory document in this set. Most other tracked types sit in the 58–87% prevalence range; nothing besides D.1 approaches ubiquity.

The naive-vs-corrected gap is where the classifier's unevenness shows up directly:

- **D.2: 85.5% naive → 63.1% corrected [55.4%, 69.5%]** — the largest correction in the whole analysis, and in the direction you'd expect from D.2's precision problem: a lot of what the classifier calls "D.2" is actually something else (mostly misclassified "Other" pages).
- **Testimonial labour: 78.9% naive → 87.2% corrected [83.5%, 90.9%]** — moves the *other* way, because its low recall means true instances were being missed and hidden inside "Other"; correcting for that recovers them.

Typical multiplicity when present ranges from ~1.3 (Approval notice, NAMA agreement — usually a single instance) up to ~3.6 (Testimonial labour) and ~2.8 (Testimonial medical form) — these two also show the largest count corrections, consistent with their recall problems.

## 2. Co-occurrence

*[co_occurrence_heatmap.png](co_occurrence_heatmap.png)*

**A confound had to be controlled first.** The raw pairwise association came back with all 45 pairs positive — not a plausible result on its face. The cause: dossiers vary hugely in overall "richness" (1 to 11 of the 11 possible types present, mean 9), so a more-complete dossier shows elevated presence for *every* type at once, inflating every pair's raw association uniformly. Controlled via a Mantel-Haenszel richness-stratified estimator (dossiers split into 4 strata by how many *other* types they contain, standard epidemiological technique for exactly this kind of confound).

**After control, every pair is still positive** — the confound explained part of the signal (weaker pairs roughly halved, e.g. D.2–Judicial background check: 0.60 → 0.31 log-odds), but nothing flipped to negative. Read together with Occurrence, this is a coherent picture: these 10 tracked types function as parts of one largely-standard "complete case" packet rather than having substitutable/competing pairs — plausible for a bureaucratic process where a successfully-processed case needed most of the same forms. (The one clear candidate for a substitution pair — medical form vs. letter — was already merged into a single type before this analysis, on your instruction, which may be part of why none turned up here.)

**D.1 is the strongest co-occurring partner with everything** (log-odds 2.0–3.9), consistent with its near-universal prevalence. The weakest-but-still-positive associations cluster around D.2, Judicial background check, and Approval notice — notably, the same types with the worst classifier reliability, so these numbers deserve extra caution even after correction.

## 3. Dispersion

*[dispersion_naive_vs_corrected.png](dispersion_naive_vs_corrected.png)*

Every type is dramatically more contiguous (filed as one bundle) than a random arrangement would produce — z-scores in the hundreds against the exact closed-form runs-test null, for all 10 types. (This calculation was validated against a hand-worked synthetic example before trusting the real output, given how large the numbers are.) This says the corpus's document instances are essentially never scattered arbitrarily through a dossier — same-type documents are consistently kept together, whether that reflects original filing practice, archival practice, or both.

**Relative ranking**: D.1 is the most rigidly clustered (contiguity score 0.17 [0.13, 0.26], on a 0=always-one-block to 1=maximally-scattered scale), while Approval notice (0.79) and Report of selection (0.71) sit at the loose end — still far more clustered than chance, just less rigidly so than D.1.

**Where correction matters most**: the two weakest-recall types again — **Testimonial labour (naive 0.30 → corrected 0.70) and Testimonial medical form (0.40 → 0.70)**. The instances recovered by the correction (previously hidden inside "Other") sit in different positions than the naively-detected cluster, revealing genuinely more scattered true patterns than the raw predictions suggested. This is the clearest demonstration in the whole analysis of why the correction isn't a formality — for these two types, the naive answer would have been substantively wrong, not just imprecise.

## 4. Order

*[order_naive_vs_corrected.png](order_naive_vs_corrected.png)*

**D.1 anchors the start** (mean normalized position 0.22, Bradley-Terry rank far ahead of everything else) and **D.2 anchors the end** (0.61, latest-ranked). This directly confirms the example that motivated this whole line of analysis: **P(D.1 before D.2) = 0.92, 94% HDI [0.89, 0.94]**.

Between the anchors: Approval notice and DM.1 come early; NAMA agreement, Testimonial medical form, Registration card, Report of selection, and Judicial background check cluster in the middle-to-late zone with mostly overlapping credible intervals — no sharp distinctions among them. Testimonial labour has the widest uncertainty band of any type (crosses zero), meaning its position is genuinely inconsistent across dossiers, not just poorly estimated.

**One counterintuitive result worth flagging for domain input**: Approval notice ranks among the *earliest* documents, which is surprising if it represents final case approval (you'd expect that at the end). Possibilities: it could be an intermediate eligibility approval rather than final sign-off, or reflect an archival-filing convention rather than original processing order. Not something I can resolve from this data alone — worth checking against what the form actually represents.

*(Before trusting the Bradley-Terry model on real data, its sign convention was checked against synthetic data with a known true ranking — it caught a bug in the synthetic-data generator on the first attempt, which was fixed and reconfirmed before running on real data. Rerunnable via `python3 scripts/dossier_composition/order.py --self-test`, also wired as a Makefile prerequisite of the `order` target.)*

## Putting the four pieces together

- **D.1 functions as the anchor document**: present in nearly every dossier, comes first, most tightly clustered, and the strongest co-occurring partner with every other type — consistent with it being a foundational intake form.
- **D.2 functions as a closing document**: reliably last when present, positively associated with everything but more weakly than D.1, and it's also the type whose occurrence rate needed the biggest correction (precision problem). Interpret D.2's specific numbers with the most caution of any tracked type.
- **The weak-classification types recur as the ones needing the biggest corrections, across independent analyses**: D.2 (occurrence), Testimonial labour and Testimonial medical form (occurrence and dispersion both), Judicial background check and Approval notice (co-occurrence). This cross-cutting pattern — the same handful of types showing the largest naive-vs-corrected gaps in every section — is itself a useful diagnostic: these are exactly the types where the incoming better-performing model's predictions are most likely to change the substantive answer, not just tighten the uncertainty.
- **The 10 tracked types read as components of one largely-standard "complete case" packet** with a fairly stable typical order (D.1 → middle cluster → D.2) rather than a set of interchangeable or competing forms — no evidence of any pair substituting for another, even after confound control.

## Limitations

- The confusion matrix is estimated from only 307 test pages; some types have thin support even after merging (Approval notice test n=7, Testimonial medical n=31) — hierarchical pooling mitigates but doesn't eliminate the resulting uncertainty, which does show up as wider corrected intervals for those types throughout.
- "Other" (everything outside the 10 tracked types) is a deliberately heterogeneous residual bucket, excluded from the co-occurrence, dispersion, and order rankings — it still acts correctly as an "interrupter" in dispersion/order calculations, just isn't reported on as its own type.
- Several approximations trade some statistical purity for tractability: Mantel-Haenszel stratification approximately (not perfectly) controls the richness confound; position statistics pool instances across dossiers as if independent; pairwise order uses a Normal approximation to the Bradley-Terry likelihood rather than full MCMC per imputation. All are standard, well-understood approximations, not ad hoc shortcuts — but worth knowing they're there.
- This analysis describes patterns in the (corrected) *digitized and predicted* record. It cannot on its own distinguish original bureaucratic filing order from disturbance introduced during archival handling or digitization — the dispersion and order results are evidence about the surviving record, not directly about 1950s-60s office practice, though the two are presumably related.

## Rerunning

```
make -C scripts/dossier_composition all \
    PREDICTIONS=data/predictions-new-model.tsv \
    TEST_PREDICTIONS=runs/new_model/test_predictions.tsv
```

Code: [scripts/dossier_composition/](../../scripts/dossier_composition/). Full tables: `occurrence_summary.csv`, `co_occurrence_summary.csv`, `dispersion_summary.csv`, `order_summary.csv`, `order_pairwise.csv` (all in this directory).
