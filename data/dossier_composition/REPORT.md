# Dossier composition analysis

**What this covers:** occurrence, co-occurrence, dispersion, and order of the 11 tracked document types across the full corpus of 3,343 dossiers, predicted by the `pipeline_latefusion_check` pipeline (`knn(DINOv2-S+mpnet-multilingual)` for start_page → `late-fusion(EfficientNet-B0+BERT)` for document_type).

**Status:** this run uses the current best pipeline (document-type macro F1 0.80, accuracy 97.1%, both end-to-end against the model's own predicted segmentation; start-page macro F1 0.95). The entire analysis reruns against new predictions with no code changes (`make -C scripts/dossier_composition all PREDICTIONS=... TEST_PREDICTIONS=...`) — treat the numbers below as the current best answer, not final.

## Why this needed a correction layer, not just counting

This classifier's recall is fairly even across types — from D.1 (99.1%) down to a still-solid Report of selection and medical officers (92.6%). The larger remaining unevenness is on *precision*: Approval notice (80.7%) and D.2 (90.7%) are the two types most often over-predicted, which inflates their naive prevalence (see Occurrence, below). Naive counts on predicted labels are still systematically off for these types, just less so and less broadly than with a weaker classifier.

**Shared method across all four analyses:** a Bayesian confusion matrix (hierarchical Dirichlet-multinomial, partially pooled across types) fit from a 338-page held-out test set, used to multiply-impute each predicted document instance's plausible "true" type (200 imputations, each also resampling the confusion matrix's own posterior). Every statistic below is computed on both the raw predicted labels ("naive") and pooled across the 200 imputations ("corrected"), so the gap between them shows exactly how much the classifier's unevenness would have distorted a naive reading. Full detail: [confusion_matrix.py](../../scripts/dossier_composition/confusion_matrix.py), [imputation.py](../../scripts/dossier_composition/imputation.py).

Recall per tracked type (posterior mean, from the confusion matrix):

| type | recall | type | recall |
|---|---|---|---|
| D.1 | 0.991 | Approval notice | 0.959 |
| D.2 | 0.982 | Testimonial medical form (Medical & Health Documents) | 0.932 |
| Testimonial medical letter (Medical & Health Documents) | 0.981 | DM.1 | 0.930 |
| Testimonial labour (Qualification & Employment Proof) | 0.974 | NAMA agreement | 0.930 |
| Registration card | 0.964 | Report of selection and medical officers | 0.926 |
| Judicial and political background check | 0.964 | | |

Recall is now solid and fairly even across the board — the remaining unevenness worth watching is *precision*: Approval notice's and D.2's raw prevalence turned out to be somewhat inflated by misclassified "Other" pages (see Occurrence, below).

## 1. Occurrence

*[occurrence_naive_vs_corrected.png](occurrence_naive_vs_corrected.png)*

D.1 is near-universal (94.8% of dossiers, [94.0%, 95.1%]) — the closest thing to a mandatory document in this set. Most other tracked types sit in the 55–92% prevalence range.

The naive-vs-corrected gap now mostly shows the classifier's *precision* unevenness, not recall:

- **Approval notice: 71.9% naive → 60.7% corrected [48.5%, 68.7%]** — the largest correction in the whole analysis, in the direction its precision problem predicts: much of what the classifier calls "Approval notice" is actually misclassified "Other" pages.
- **D.2: 89.6% naive → 83.2% corrected [77.6%, 87.7%]** — the same pattern, more mildly.

With recall now solid across the board, no type shows the opposite pattern (recall recovering hidden instances) by more than a percentage point or two — every other type's naive and corrected prevalence agree closely.

## 2. Co-occurrence

*[co_occurrence_heatmap.png](co_occurrence_heatmap.png)*

**A confound had to be controlled first.** The raw pairwise association came back with all 55 pairs positive — not a plausible result on its face. The cause: dossiers vary hugely in overall "richness" (1–12 of the 12 possible types present, mean 9.6), so a more-complete dossier shows elevated presence for *every* type at once, inflating every pair's raw association uniformly. Controlled via a Mantel-Haenszel richness-stratified estimator (dossiers split into 4 strata by how many *other* types they contain, variance via Robins–Breslow–Greenland — standard epidemiological technique for exactly this kind of confound).

**After control, every pair is still positive** — the confound explained part of the raw signal (weaker pairs shrink substantially, e.g. D.2–Judicial background check: 1.29 → 0.53 log-odds), but nothing flipped to negative. Read together with Occurrence, this is a coherent picture: these 11 tracked types function as parts of one largely-standard "complete case" packet rather than having substitutable/competing pairs. D.1–D.2 specifically: log-odds 3.31 [2.38, 4.11], among the strongest pairs in the corpus.

**D.1 is the strongest co-occurring partner with everything** (log-odds 1.55–4.06 across its 10 partners). The weakest-but-still-positive associations cluster around Testimonial medical form, Judicial and political background check, and Testimonial medical letter. With this classifier's much more even reliability, that clustering is no longer purely a residual artifact for all three: Approval notice's and D.2's weak precision still explain part of it, but Judicial background check's recall and precision are both solid now — its weak co-occurrence looks like a genuine substantive signal rather than a correction gap.

## 3. Dispersion

*[dispersion_naive_vs_corrected.png](dispersion_naive_vs_corrected.png)*

Every type is dramatically more contiguous (filed as one bundle) than a random arrangement would produce — z-scores in the hundreds against the exact closed-form runs-test null, for all 11 types. (This calculation was validated against a hand-worked synthetic example before trusting the real output, given how large the numbers are.) This says the corpus's document instances are essentially never scattered arbitrarily through a dossier — same-type documents are consistently kept together, whether that reflects original filing practice, archival practice, or both.

**Relative ranking**: Judicial and political background check is now the most rigidly clustered (contiguity score 0.06 [0.04, 0.13]), while Approval notice (0.57) sits at the loose end — still far more clustered than chance, just less rigidly so. Which type anchors each end is itself sensitive to the classifier: a type whose weak instances get merged into "Other" naively looks artificially tight until correction spreads it back out, so this pairing is worth re-checking whenever the underlying model changes.

**Where correction matters most**: not simply the weakest-recall types (recall is fairly even now) but the types whose predicted instances get relocated the most once segmentation and type are jointly corrected — **NAMA agreement (0.17 → 0.31) and Report of selection and medical officers (0.19 → 0.30)**. Instances recovered or reassigned sit in different positions than the naively-detected cluster, revealing genuinely more scattered true patterns for these two. This is the clearest demonstration in the whole analysis of why the correction isn't a formality — for these two types, the naive answer would have been substantively wrong, not just imprecise.

## 4. Order

*[order_naive_vs_corrected.png](order_naive_vs_corrected.png)*

**D.1 anchors the start** (mean normalized position 0.23) and **D.2 anchors the end** (0.60), fit with a Bradley-Terry paired-comparison model on which type's first occurrence comes earlier, for every dossier where a pair co-occurs. This directly confirms the example that motivated this whole line of analysis: **P(D.1 before D.2) = 90.7%, 94% HDI [88.8%, 91.6%]**.

Between the anchors, the remaining types rank (earliest to latest, mostly overlapping intervals, no sharp distinctions among neighbors): DM.1, NAMA agreement, Registration card, Report of selection and medical officers, Judicial and political background check, Testimonial labour, Testimonial medical letter, and Testimonial medical form. Testimonial medical form has the widest uncertainty band of any type, meaning its position is genuinely inconsistent across dossiers, not just poorly estimated.

**One counterintuitive result worth flagging for domain input**: Approval notice ranks among the *earliest* documents (mean position 0.37), which is surprising if it represents final case approval (you'd expect that at the end). Possibilities: it could be an intermediate eligibility approval rather than final sign-off, or reflect an archival-filing convention rather than original processing order. Not something resolvable from this data alone — worth checking against what the form actually represents.

*(Before trusting the Bradley-Terry model on real data, its sign convention was checked against synthetic data with a known true ranking — it caught a bug in the synthetic-data generator on the first attempt, which was fixed and reconfirmed before running on real data. Rerunnable via `python3 scripts/dossier_composition/order.py --self-test`, also wired as a Makefile prerequisite of the `order` target.)*

## Putting the four pieces together

- **D.1 functions as the anchor document**: present in nearly every dossier, comes first, is the most tightly clustered type, and the strongest co-occurring partner with everything else — consistent with it being a foundational intake form.
- **D.2 functions as a closing document**: reliably last when present, positively associated with everything but more weakly than D.1. Its occurrence rate still needs a real correction (89.6% → 83.2%) — second only to Approval notice (71.9% → 60.7%), the type that now needs the single biggest correction. Interpret both types' numbers with more caution than the rest.
- **A cross-cutting diagnostic**: Approval notice and D.2 — the two weakest-precision types — keep showing up as needing the biggest naive-vs-corrected corrections in occurrence and co-occurrence specifically; Testimonial labour and DM.1 show the same pattern in dispersion instead. No single type is unreliable across every analysis at once with this classifier — the earlier, weaker model produced a more uniform "usual suspects" list than this one does.
- **The 11 tracked types read as components of one largely-standard "complete case" packet** with a fairly stable typical order (D.1 → middle cluster → D.2) rather than a set of interchangeable or competing forms — no evidence of any pair substituting for another, even after confound control.

## Limitations

- The confusion matrix is estimated from only 338 test pages; some types have thin support even after merging (Approval notice test n=9, Testimonial medical n=5) — hierarchical pooling mitigates but doesn't eliminate the resulting uncertainty, which does show up as wider corrected intervals for those types throughout.
- "Other" (everything outside the 11 tracked types) is a deliberately heterogeneous residual bucket, excluded from the co-occurrence, dispersion, and order rankings — it still acts correctly as an "interrupter" in dispersion/order calculations, just isn't reported on as its own type.
- Several approximations trade some statistical purity for tractability: Mantel-Haenszel stratification approximately (not perfectly) controls the richness confound; position statistics pool instances across dossiers as if independent; pairwise order uses a Normal approximation to the Bradley-Terry likelihood rather than full MCMC per imputation. All are standard, well-understood approximations, not ad hoc shortcuts — but worth knowing they're there.
- This analysis describes patterns in the (corrected) *digitized and predicted* record. It cannot on its own distinguish original bureaucratic filing order from disturbance introduced during archival handling or digitization — the dispersion and order results are evidence about the surviving record, not directly about 1950s-60s office practice, though the two are presumably related.

## References

1. Dawid, A. P., & Skene, A. M. (1979). Maximum likelihood estimation of observer error-rates using the EM algorithm. *Journal of the Royal Statistical Society: Series C*, 28(1), 20–28. — estimating a confusion/error-rate matrix from noisy classifier or rater labels, the basis for the correction used throughout.
2. Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press. — hierarchical/partial-pooling modeling generally, used for the confusion matrix and throughout the wider project.
3. Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, 15(1), 1593–1623. — the MCMC sampler used to fit the confusion matrix.
4. Salvatier, J., Wiecki, T. V., & Fonnesbeck, C. (2016). Probabilistic programming in Python using PyMC3. *PeerJ Computer Science*, 2, e55.
5. Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys*. John Wiley & Sons. — the multiple-imputation framework used to propagate classifier uncertainty into every downstream statistic.
6. Mantel, N., & Haenszel, W. (1959). Statistical aspects of the analysis of data from retrospective studies of disease. *Journal of the National Cancer Institute*, 22(4), 719–748. — the stratified estimator used to control the dossier-richness confound in Co-occurrence.
7. Robins, J., Breslow, N., & Greenland, S. (1986). Estimators of the Mantel-Haenszel variance consistent in both sparse data and large-strata limiting models. *Biometrics*, 42(2), 311–323. — the variance estimator used alongside the Mantel-Haenszel odds ratio.
8. Wald, A., & Wolfowitz, J. (1940). On a test whether two samples are from the same population. *Annals of Mathematical Statistics*, 11(2), 147–162. — the runs-test null distribution used in Dispersion to judge dispersion against random placement.
9. Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika*, 39(3/4), 324–345. — the paired-comparison ranking model used in Order to derive a typical sequence order.

## Rerunning

```
make -C scripts/dossier_composition all \
    PREDICTIONS=data/predictions-new-model.tsv \
    TEST_PREDICTIONS=runs/new_model/test_predictions.tsv
```

Code: [scripts/dossier_composition/](../../scripts/dossier_composition/). Full tables: `occurrence_summary.csv`, `co_occurrence_summary.csv`, `dispersion_summary.csv`, `order_summary.csv`, `order_pairwise.csv` (all in this directory).
