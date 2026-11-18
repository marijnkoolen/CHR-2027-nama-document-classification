# Correcting for the computer's mistakes

### How we use a confusion matrix, Bayesian modelling and multiple imputation to get honest statistics from automatically classified dossiers

*A plain-language explanation. No statistics background needed.*

---

## The short version

We did not read the roughly 115,000 pages in the migration dossiers ourselves. A computer model (a *classifier*) did. For every page it decided two things:

1. **Does a new document start on this page?** This splits a dossier into separate documents.
2. **What type of document is this?** For example a D.1 form, a NAMA agreement or a Registration card.

The classifier is good, but it is not perfect, and **it is better at some document types than at others**. If we simply counted what it predicted, some numbers would come out too high, others too low, and we would not know which.

So we do four things:

1. **Measure the mistakes.** We check the classifier against a sample of pages whose correct answer a person has recorded. That gives a table of who-gets-mistaken-for-whom: the **confusion matrix**.
2. **Be honest about how well we know those mistakes.** The check sample is small, so we use a **Bayesian model** to estimate not one error table but a whole range of plausible ones.
3. **Fill in plausible true labels.** For every predicted document we "roll a weighted die" to choose what it most plausibly really is, given what the classifier said. This is **imputation**.
4. **Repeat many times and compare.** We build many plausible versions of the corpus, compute our statistics on each, and report the typical value plus a range. This is **multiple imputation**.

The result: statistics that are **corrected** for the classifier's known biases and come with an **honest range** of uncertainty, instead of numbers that look precise but are quietly wrong.

---

## 1. The problem: an assistant who makes uneven mistakes

Imagine you hire an assistant to sort 115,000 archive pages into piles by document type. They are fast and usually right. But:

- Some forms are distinctive and they never get them wrong (the D.1 form, for example).
- Some look like other documents, and they sometimes put them on the wrong pile.
- There is also a large mixed pile called **"Other"**: everything that isn't one of the 11 types we track. Because that pile is so large, even a small slip rate from it adds a noticeable number of wrong pages to the smaller piles.

Now suppose you ask: *"In what share of dossiers does an Approval notice appear?"* If you count the assistant's Approval notice pile at face value, you also count every page they wrongly put there. And the size of that error is **different for every document type**. A uniform small error would be tolerable. An uneven one distorts comparisons between types, which is exactly what our analyses are about.

The same applies to **where documents begin**. If the assistant misses the first page of a new document, two documents get merged into one. If they wrongly think a new document starts, one document gets split into two. Both change the counts.

**What this procedure is for:** taking the classifier's predictions, together with what we know about how and how often it errs, and producing statistics about the dossiers as they *really* are, along with an honest statement of how sure we can be.

---

## 2. Step one: measuring the mistakes (the confusion matrix)

To find out how the assistant errs, you take a sample of pages, have an expert record the correct answer for each, and compare. In our project this is the **test set**: 338 pages that the classifier never saw while it was being trained, each with a correct label added by a person.

The comparison is laid out as a table called a **confusion matrix**:

- each **row** is what the page *really* is;
- each **column** is what the classifier *said* it is;
- each cell counts how many pages fall into that combination.

Pages on the diagonal (row and column the same) were classified correctly. Everything off the diagonal is a mistake, and the table shows *which* mistake.

Here is an excerpt from the current classifier's test results (document types):

| Really is ↓ / Classifier said → | Approval notice | D.2 | DM.1 | NAMA agreement | Report of selection | Testimonial labour | Other |
|---|---|---|---|---|---|---|---|
| **Approval notice** (9 pages) | **9** | 0 | 0 | 0 | 0 | 0 | 0 |
| **DM.1** (33 pages) | 0 | 0 | **31** | 0 | 0 | 0 | 2 |
| **NAMA agreement** (19 pages) | 0 | 0 | 1 | **18** | 0 | 0 | 0 |
| **Report of selection…** (18 pages) | 1 | 0 | 0 | 0 | **17** | 0 | 0 |
| **Testimonial labour** (52 pages) | 0 | 0 | 0 | 0 | 0 | **51** | 1 |
| **Other** (93 pages) | 1 | 2 | 0 | 0 | 0 | 3 | **87** |

How to read it: of the 93 test pages that were really "Other", 87 were correctly labelled "Other", but 1 was called an Approval notice, 2 were called D.2 and 3 were called Testimonial labour.

And for the start-of-a-new-document question:

| Really is ↓ / Classifier said → | not a start | start |
|---|---|---|
| **not a start** (100 pages) | **92** | 8 |
| **start of a new document** (238 pages) | 5 | **233** |

So in the test set the classifier missed 5 real document starts (which would merge documents) and invented 8 starts that weren't there (which would split documents).

---

## 3. Step two: being honest about how well we know the mistakes (the Bayesian part)

The test set is small. Some types appear only a handful of times: Approval notice 9 times, Testimonial medical form only 5 times. That raises a real question.

> The classifier got all 9 Approval notices right. Does that mean it **never** gets them wrong?

Common sense says no. With only 9 tries, a perfect score is also quite possible for an assistant who is right "only" 85% of the time. A larger sample might well show a slip or two. Taking 9 out of 9 as "100% reliable" would be overconfident.

**Bayesian modelling** is a way of reasoning that makes this common sense precise. It works in three moves.

**a. Start with a sensible expectation (the "prior").**
Before looking at the test results, we assume something modest: *the classifier is usually right, and when it is wrong, it is most likely to pick a label that is common overall.* In the code this starting expectation puts 70% of the weight on "correct" and spreads the remaining 30% across labels according to how often the classifier uses each label.

**b. Let the data update that expectation.**
The test results are then combined with the starting expectation. Where there is plenty of evidence (93 "Other" pages, 52 Testimonial labour pages), the evidence dominates and the starting expectation hardly matters. Where there is little evidence (5 or 9 pages), the estimate leans more on the starting expectation and stays appropriately cautious.

**c. Borrow strength across types.**
How much weight should the starting expectation get? Rather than picking a number by hand, the model *learns it from all twelve types together*. Think of judging a new restaurant that has only three reviews: you would partly go by those three reviews and partly by how restaurants of that kind usually do. The model does the same for thinly-tested document types: it lets them borrow from the overall pattern across all types. Statisticians call this **partial pooling**, or a **hierarchical** model.

**The result is not one error table but thousands.** The computer generates a large collection (6,000) of plausible confusion matrices. Each is consistent with the test results and the starting expectation, and tables that fit the evidence better appear more often. Together they express both *what we think the error rates are* and *how unsure we are about them*.

For example, the average of these plausible tables puts the chance that a real Approval notice is recognised correctly at about 96%, not 100%. Individual tables in the collection range somewhat above and below that.

---

## 4. Step three: turning the question around

The confusion matrix answers: *"If a page **really is** an Approval notice, what will the classifier say?"*

But in the 115,000-page corpus we face the reverse question: *"The classifier **says** Approval notice. What is the page really?"*

These are not the same, and the difference matters. The reverse answer also depends on **how common each type is**. Recall that of the 93 "Other" test pages, 1 was called an Approval notice. That is a slip rate of only about 1 in 100. But "Other" pages are roughly ten times as common as Approval notices, so even that small slip rate adds a noticeable number of false "Approval notices" to the pile. The same goes for the one Report of selection that was mistaken for one.

Combining the error table with how common each type is (using a rule known as **Bayes' rule**) gives, for the current classifier:

> **When the classifier says "Approval notice"**, the page is
> - really an Approval notice about **82%** of the time,
> - really "Other" about **8.5%** of the time,
> - really a Report of selection and medical officers about **8.5%** of the time.

In other words, of every 12 pages labelled Approval notice, roughly 10 really are one and about 2 are something else. For other labels the picture is different:

| When the classifier says… | it really is that type about… | most common alternative |
|---|---|---|
| D.1 | 99% | (spread thinly) |
| NAMA agreement | 99% | (spread thinly) |
| DM.1 | 96% | NAMA agreement, 3% |
| Testimonial labour | 93% | Other, 5.5% |
| D.2 | 91% | Other, 8% |
| Approval notice | 82% | Other / Report of selection, 8.5% each |
| Other | 95% | DM.1, 2.3% |

The last row matters too. A small share of pages the classifier calls "Other" are really tracked types (for example DM.1). Correcting for this *raises* the estimated counts of those types. **The correction can push numbers up as well as down.**

The same reversal is done for document starts. With the current classifier:

- a page marked as **the start of a new document** really is one about **96%** of the time;
- a page marked as **a continuation** is really the start of a new document about **5.5%** of the time.

---

## 5. Step four: filling in plausible true labels (imputation)

We now know, for every kind of prediction, the chances of what is really there. **Imputation** means filling in a plausible value where the true value is unknown. Here, it means choosing a plausible *true* label for each predicted page or document.

We do this by chance, in proportion to those probabilities. Picture a weighted die for each document the classifier labelled "Approval notice": it lands on "Approval notice" 82% of the time, on "Other" 8.5% of the time and on "Report of selection" 8.5% of the time. Roll it once for every such document in the corpus. A document labelled "D.1" gets a die that lands on "D.1" about 99% of the time, and so on.

This happens in two layers, in this order:

1. **Document boundaries first.** For every page, roll a die for "is this really the start of a new document?" using the start-page probabilities. Then rebuild the documents from those corrected starts. Some documents get merged and some get split, so the number of documents per dossier can change.
2. **Document types second.** For each rebuilt document, take the type the classifier gave most of its pages (a majority vote), then roll that type's die to choose a plausible true type.

The outcome is one complete, **plausible version of the corpus**. Every document has a boundary and a type, and the patterns of error we measured have been accounted for on average.

---

## 6. Step five: repeat many times and summarise (multiple imputation)

One roll of the dice gives one plausible version. A different roll gives a slightly different one. Neither is "the truth". The truth is somewhere in the spread. So we:

1. **Repeat the whole filling-in step many times**: 200 times for the dossier composition analyses, and 20 times for the heavier dossier size models (each of those takes about 25 minutes of computing on its own).
2. **Use a different plausible error table each time**, drawn from the 6,000 from step two. This means the spread reflects two kinds of uncertainty: the luck of the dice (*which* documents were misclassified) and our uncertainty about the error rates themselves (*how often* the classifier errs).
3. **Compute the statistic we care about on each version**, for example "share of dossiers that contain an Approval notice".
4. **Summarise across all versions.** The average is our **corrected estimate**. The range covering the middle 94% of versions (from the 3rd to the 97th percentile) is our **uncertainty interval**. In 94 out of 100 plausible versions of the corpus, the value fell inside that range.

We also compute every statistic once on the classifier's raw predictions, the **naive** value. Comparing naive and corrected shows how much the classifier's mistakes would have distorted the picture.

### The whole procedure at a glance

```
  Test set (338 pages, correct answers recorded by a person)
        │
        ▼
  ① Confusion matrices: who gets mistaken for whom
     (document type, and start of a new document)
        │
        ▼
  ② Bayesian model → 6,000 plausible error tables
        │
        │        Classifier's predictions for ~115,000 pages
        │                       │
        ▼                       ▼
  ③ Turn around: "classifier says X → what is it really?"
        │
        ▼
  ④ Fill in one plausible version of the corpus
     (a) roll dice for document starts → rebuild documents
     (b) roll dice for each document's type
        │
        ▼
  ⑤ Compute the statistic on this version ──┐
        │                                   │ repeat 200×
        └──────── next error table ◄────────┘ (20× for size models)
        │
        ▼
  Corrected estimate (average) + 94% range
        compared with the naive (uncorrected) value
```

---

## 7. What it does to the results: examples

### Dossier composition (`scripts/dossier_composition/`)

These analyses describe what dossiers are made of: how often each document type appears (**occurrence**), which types appear together (**co-occurrence**), whether documents of one type sit together or are scattered (**dispersion**), and in what order types typically appear (**order**). Every one is computed both naively and corrected.

Share of dossiers containing at least one document of a type (current classifier, all 3,343 dossiers):

| Document type | Naive | Corrected | 94% range |
|---|---|---|---|
| D.1 | 95.2% | 94.8% | 94.0 – 95.1% |
| NAMA agreement | 91.2% | 88.3% | 84.1 – 90.2% |
| DM.1 | 83.4% | 84.6% | 82.3 – 87.2% |
| D.2 | 89.6% | 83.2% | 77.6 – 87.7% |
| Approval notice | 71.9% | 60.7% | 48.5 – 68.7% |
| Testimonial medical form | 40.5% | 41.0% | 38.2 – 46.7% |

What to take from this:

- **Reliable types barely move.** D.1 is recognised almost perfectly, so its naive and corrected values are nearly the same, and the range is narrow.
- **Some types move a lot.** Approval notice drops by about 11 percentage points, and the naive value (71.9%) lies *outside* the corrected range. Reported naively, we would have overstated how common Approval notices are. D.2 drops by about 6 points for the same reason: some "Other" pages get mislabelled as D.2.
- **Some types move up.** DM.1 rises slightly, because a few DM.1 documents were hidden in the "Other" pile.
- **The ranges differ.** Approval notice has a wide range (about 20 points) because it is thinly represented in the test set and mixed up with common types. That width is honest information: it tells the reader how firmly a claim about Approval notices can be made.

### Dossier size and family composition (`scripts/dossier_size_model/`)

These models ask how the number of documents in a dossier relates to who is in the family (children under 16, 16–17 year-olds, adults) and to the travel year, including whether things changed around 1956. The classifier's output enters in two places, and each gets the correction it needs.

**Total number of documents per dossier.** This depends only on *where documents begin*, not on what type they are, so only the start-of-document correction is needed. Here we use a quicker shortcut (see below): each page counts as a *fraction* of a document start. A page the classifier marks as a start counts as about 0.96; a page it marks as a continuation counts as about 0.055. The first page of a dossier always counts as 1. Summed per dossier, this gives an expected corrected count. For the 1,307 dossiers in this analysis, the average went from 24.5 to 24.2 documents, and 285 dossiers shifted by more than half a document. All models were then refitted on the corrected counts. The ranking of the three competing explanations for the 1956 change stayed the same.

**Number of documents of each type per family-member group.** This depends on both boundaries and types, so it gets the full two-layer multiple imputation from section 6. For each of 20 plausible versions of the corpus, the *entire* statistical model is refitted, and the 20 sets of results are then pooled. This takes enough computing power that it runs on a GPU server.

### The shortcut vs. the full procedure

The dossier size scripts offer two modes:

- **Shortcut ("point" mode).** Use only the *average* error table, and give each document fractional counts instead of rolling dice ("0.82 of an Approval notice, 0.085 of a Report of selection…"). This is fast, runs on an ordinary computer, and **corrects the average bias**. But it treats the error rates as exactly known and skips the dice, so its **ranges come out too narrow**. It is good for a quick check of whether the correction changes the story.
- **Full multiple imputation ("mi" mode).** Everything described in sections 5 and 6. It is slower, but it **corrects the bias and gives honest ranges**. Where the full version is available, it is the one to trust.

---

## 8. Why it matters

- **It prevents false findings.** Without the correction, we would have reported Approval notices in about 72% of dossiers. The best corrected estimate is about 61%, and the naive figure lies outside the plausible range. Differences between document types, between periods, or between family types can be partly or entirely produced by the classifier rather than the archive. The correction separates the two.
- **It makes comparisons fair.** Because the classifier's reliability differs by type, raw counts are not comparable across types. After correction, every type has been adjusted by its own error pattern.
- **It tells us how sure we can be.** A number without a range invites over-reading. The ranges show where the evidence is firm (D.1) and where it is thin (Approval notice), so claims in the paper can be worded with the right strength.
- **It is transparent and checkable.** Every result is reported both naively and corrected. Readers can see exactly how much the correction changed, and judge for themselves.
- **It keeps working when the classifier improves.** When a better classifier becomes available, we re-measure its mistakes on the test set and rerun the pipeline. No code changes are needed. A better classifier simply produces a smaller gap between naive and corrected, and narrower ranges.

---

## 9. What it cannot do (assumptions and limits)

The correction is only as good as the information it rests on. It assumes:

- **The test set is representative.** The classifier is assumed to make the same kinds of mistakes, at the same rates, across the whole corpus as in the 338 test pages. How common each type is in the test set is also used in step three. If some dossiers (say, from a particular year or office) are harder to classify than the test pages suggest, the correction will not fully capture that.
- **Mistakes are independent.** Each page or document is corrected on its own, without considering its neighbours or the rest of its dossier. In reality a person might use context ("there's already a D.1 here, so this is probably not another one"). The correction does not.
- **It can only fix mistakes it has seen.** Kinds of confusion that never occurred in the test set get only the small default allowance from the starting expectation. Rare but real kinds of mistake may be underestimated.
- **Thin evidence stays thin.** Types with very few test examples get wide ranges. The method reports that uncertainty but cannot remove it. Only more labelled test pages can.
- **It describes the surviving record.** The correction is about classification errors. It cannot tell us whether pages were lost, reordered or misfiled in the archive or during digitisation.

These limits do not make the corrected numbers unreliable. They are the reason the method reports ranges, and the reason naive and corrected values are shown side by side.

---

## 10. Glossary

| Term | Plain meaning |
|---|---|
| **Classifier** | The computer model that labels each page (document type; start of a new document or not). |
| **Prediction / predicted label** | What the classifier said about a page. |
| **Test set** | Pages with a correct answer recorded by a person, which the classifier did not learn from. Used to measure its mistakes. |
| **Confusion matrix** | A table counting, for each true type, what the classifier said. It shows who gets mistaken for whom. |
| **Naive** | Computed directly from the classifier's predictions, as if they were all correct. |
| **Corrected** | Computed after accounting for the classifier's measured mistakes. |
| **Bayesian modelling** | Reasoning that starts from a sensible expectation and updates it with evidence. It produces a range of plausible answers rather than a single one. |
| **Prior** | The starting expectation before looking at the evidence. Here: "usually right; when wrong, probably a common label". |
| **Posterior** | The updated range of plausible answers after combining the prior with the evidence. Here: 6,000 plausible error tables. |
| **Partial pooling / hierarchical model** | Letting types with little evidence borrow from the overall pattern across all types. |
| **Base rate** | How common something is overall. It matters when turning "true → predicted" into "predicted → true". |
| **Bayes' rule** | The arithmetic for that turning-around, combining error rates with base rates. |
| **Imputation** | Filling in a plausible value where the true value is unknown. |
| **Multiple imputation** | Doing that filling-in many times, analysing each version, and combining the results so the uncertainty is kept rather than hidden. |
| **94% range (interval)** | The span containing the middle 94% of results across all plausible versions. |
| **Segmentation** | Splitting a dossier's pages into separate documents, based on predicted document starts. |
| **Point mode** | The shortcut: average error table and fractional counts. Fast; corrects bias; ranges too narrow. |
| **MI mode** | The full multiple-imputation procedure. Slower; corrects bias and gives honest ranges. |

---

## 11. Where this lives in the code

| Step | File |
|---|---|
| Which document types are tracked (11 types + "Other") | [`scripts/dossier_composition/common.py`](../scripts/dossier_composition/common.py) |
| ①② Document-type confusion matrix (Bayesian model) | [`scripts/dossier_composition/confusion_matrix.py`](../scripts/dossier_composition/confusion_matrix.py) |
| ①② Start-of-document confusion matrix (Bayesian model) | [`scripts/dossier_composition/start_page_confusion_matrix.py`](../scripts/dossier_composition/start_page_confusion_matrix.py) |
| ③④ Turning around, rolling the dice, rebuilding documents | [`scripts/dossier_composition/imputation.py`](../scripts/dossier_composition/imputation.py) |
| ⑤ Composition analyses (naive vs. corrected) | `occurrence.py`, `co_occurrence.py`, `dispersion.py`, `order.py` in [`scripts/dossier_composition/`](../scripts/dossier_composition/) |
| Linking predictions to the dossier size data; shortcut and full corrections | [`scripts/dossier_size_model/doc_type_uncertainty_common.py`](../scripts/dossier_size_model/doc_type_uncertainty_common.py) |
| Corrected total-document-count models (shortcut) | [`scripts/dossier_size_model/aggregate_uncertainty.py`](../scripts/dossier_size_model/aggregate_uncertainty.py) |
| Corrected per-type models (shortcut and full MI) | `doc_type_three_groups_uncertainty.py`, `doc_type_count_vs_presence_uncertainty.py` in [`scripts/dossier_size_model/`](../scripts/dossier_size_model/) |
| Formal equations | [`docs/model_equations.tex`](model_equations.tex) |

*Numbers in this document come from the current classifier's test results and analysis outputs (`data/dossier_composition/`, `data/dossier_size_model/uncertainty/`). They will change when the pipeline is rerun with a new classifier. The procedure stays the same.*
