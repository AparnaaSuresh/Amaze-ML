# Entity Resolution Methodology Research

## Decision in one sentence

Build a **budgeted, multi-route sparse retrieval system followed by a supervised
tabular reranker and a conservative, entity-aware F<sub>0.5</sub> decision
policy**.  Treat dense/neural retrieval as a measured recall-rescue experiment,
not the default.  This is stronger than a single blocking-plus-classifier
pipeline because it makes the candidate set an explicitly measured product
objective, while retaining the character-level evidence that is most likely to
survive the challenge's name and address corruption.

This is a research conclusion, not a score guarantee.  The training/test files
are now available locally, but the modelling analyses in this document have not
yet been run; all data-dependent choices below are experiments to perform.

## 1. Scope, evidence, and hard constraints

**Decision supported.** Select an architecture for linking each deduplicated
Source-1 (S1) business to zero or more Source-2/Source-3 (S2/S3) records, using
only supplied files.  The important date-independent rules are the challenge
statement in `student_resource/README.md` and the later organiser update: the
final `candidate_pairs.tsv` is reviewed with the score and a *smaller* final
candidate set per S1 is favoured in final ranking.

This last rule changes the problem from “maximise blocking recall” to a
bi-objective problem:

```
maximise:  validation macro-F0.5 and candidate-link recall
minimise:  total candidates, mean candidates/S1, and tail (p95/p99) candidates/S1
subject to: every predicted match is a final candidate
```

No public formula says how organisers trade candidate-set size against
leaderboard score.  Therefore do **not** optimise a made-up combined score.
Maintain a Pareto frontier of runs and retain the smallest candidate set that
does not cause a material validation-score loss.

### Dataset inspection performed

The supplied archive `6ab10eb3b23ba_student_resource.zip` was unpacked into the
Git-ignored local `student_resource/dataset/` directory.  A streaming profile
found:

| file | rows | country counts / observation |
|---|---:|---|
| train S1 | 2,206,821 | India 883,188; US 1,323,633 |
| train S2 | 5,034,616 | India 2,017,799; US 3,016,817 |
| train S3 | 5,285,603 | India 2,115,547; US 3,170,056 |
| train ground truth | 2,206,821 | 123,247 empty labels (5.58%) |
| test S1 | 1,732,544 | France 259,452; India 809,986; US 663,106 |

Training S1 names/addresses were complete in this first pass, with mean lengths
24.0 and 52.1 characters respectively.  The ground truth match-count
distribution is important: labels have 0 through 11 matches; the modal cases
have 3--5.  This is evidence that a “one best candidate” formulation is unsafe.
Profile missingness, script/language, postal/number extraction, duplicate
normalised values, candidate recall, and target-side multiplicity before making
any country- or field-specific rule.

The full test S2/S3 scan was not completed in this initial timed profile.  The
challenge statement, rather than inference from train, is the authority that
France appears at test; test handling must be open-set.

## 2. What research supports—and what it does not

### Established results

* Exhaustive linkage is quadratic.  Blocking is therefore a necessary retrieval
  approximation, not an optional speed tweak.  The Splink documentation makes
  the practical point well: several strict, complementary blocking rules are
  generally preferable to one loose one, and their union must be deduplicated.
  [Splink blocking guide](https://moj-analytical-services.github.io/splink/demos/tutorials/03_Blocking.html)
* Fellegi--Sunter provides the foundational likelihood-ratio view: agreement
  patterns should be assessed by how much more likely they are for matches than
  nonmatches.  Its conditional-independence assumptions are too restrictive to
  be the final model here, but its discipline—rare agreements are stronger
  evidence than common ones, and decisions may abstain—is highly useful.
  [Fellegi & Sunter (1969)](https://www.cs.cornell.edu/~shmat/courses/cs6434/fellegi-sunter.pdf)
* Modern entity-matching models normally assume already-blocked labelled pairs.
  For example, DeepMatcher explicitly trains on labelled tuple pairs; it does
  not solve 1.7M-by-10M candidate generation.  [DeepMatcher repository](https://github.com/anhaidgroup/deepmatcher)
  Ditto showed that pretrained language models can improve benchmark entity
  matching, but that result is a pair-classification result, not proof that a
  cross-encoder should retrieve from ten million records.
  [Ditto paper](https://arxiv.org/abs/2004.00584)
* Dense approximate-nearest-neighbour search is technically feasible at this
  scale.  FAISS supports batched, approximate searches and indexes larger than
  RAM, with a speed/recall trade-off.  Feasible indexing does *not* establish
  that a general text embedding separates two distinct businesses with similar
  semantic names/addresses.  [FAISS documentation](https://faiss.ai/)
* Graph transitive closure can repair inconsistent pair decisions, but it can
  also propagate one false edge through a component.  Research on cluster
  editing exists, but it assumes transitivity constraints that this challenge's
  target-side semantics have not yet empirically established.
  [Transitivity-constraint study](https://arxiv.org/abs/2104.12589)

### Consequence for this challenge

The conventional proposed architecture is **not wrong**.  It fails if
“blocking” means one brittle key, if its candidate budget is unmeasured, if it
trains on easy random pairs, or if a global probability threshold turns common
name/address collisions into false merges.  A better formulation is retrieval +
reranking + decision, with each boundary measured separately.

## 3. Methods considered and rejected as the default

| Approach | Candidate recall | Precision potential | Scale | Complexity | Compute | Robustness | F0.5 potential |
|---|---|---|---|---|---|---|---|
| Exact normalised rules only | Low--Medium | High | High | Low | Low | Low | Medium |
| Fellegi--Sunter / deterministic-probabilistic only | Medium | Medium--High | Medium | Medium | Low--Medium | Medium | Medium |
| Single-key blocking + GBDT | Medium | High | High | Medium | Low | Medium | Medium |
| Multi-route sparse retrieval + GBDT | High, to measure | High | High | Medium | Low--Medium | High | **High** |
| Hybrid sparse + dense retrieval + GBDT | Potentially higher | High after rerank | Medium--High | High | Medium--High | Medium--High | High, if measured gain |
| Cross-encoder on final candidates | N/A alone | High | Medium only after tight retrieval | High | GPU preferred | Medium | Conditional |
| End-to-end Siamese/bi-encoder retrieval | Unknown | Medium | Medium--High with ANN | High | GPU preferred | Medium | Conditional |
| Graph/cluster-first resolution | Unknown | Low--High | Medium | High | Medium | Low unless constraints hold | Conditional |

These are qualitative ratings, not fabricated benchmark numbers.

**Why sparse character retrieval is the default.** Character 3--5 grams keep
signal through punctuation, suffix changes, local typos, token reordering and
some transliteration artefacts.  They also give explainable, cheap top-K
retrieval and have no dependence on external semantic knowledge.  BM25/token
retrieval complements it for rare words and long addresses.  Exact structured
keys can be exceptionally precise.  Their union catches different errors.

**Why not dense retrieval first.** A multilingual embedding might rescue semantic
or transliteration cases, but it can also retrieve “same type of business”
instead of “same business”, precisely the false merge F0.5 punishes.  It requires
model-license verification, embedding 10M records, ANN tuning, GPU/large-memory
operations, and train-ground-truth evidence that it recovers misses left by
sparse methods.  Run it only as an ablation and retain only the net Pareto gain.

**Why not graph clustering first.** S1 is already the reference set and outputs
are directed S1 -> {S2,S3}.  A graph adds a large failure surface: one erroneous
high-score bridge can merge a whole component.  Use graph consistency only as a
post-hoc diagnostic or a conservative veto after verifying target-side
one-to-one/cluster constraints in training.

## 4. Recommended architecture

### Stage A — loss-aware normalisation, never destructive replacement

Keep raw text *and multiple normalised views*; do not collapse everything to one
string.  Store whether a feature is missing and preserve numeric tokens.

* Unicode NFKC/case fold, punctuation/whitespace normalisation, `& -> and`.
* Names: a legal-suffix-free view; token-sorted view; original-order view;
  acronym/initial view; character grams.  Do not blindly remove meaningful
  short tokens.
* Addresses: tokenised and character views; extracted digit groups/house number;
  postal/PIN-like token; street/locality-like residual tokens; abbreviation map
  learned from supplied data or a manually auditable generic punctuation map.
  Never geocode or call address parsers backed by external data.
* Country: preserve raw value and use exact equality as a strong retrieval
  partition/feature, with an explicit `unknown/unseen` path.  France must flow
  through the same string rules; it must not be dropped or one-hot rejected.

An address should be represented both as a string and as imperfect components.
Component agreement (same rare postal code or house number plus street evidence)
is useful; a component conflict (different reliable numbers) is a strong
non-match feature.  Parsing errors must not erase the full-string route.

### Stage B — budgeted candidate generation

Build S2 and S3 indexes separately or retain source as a field.  For every S1,
retrieve candidates by the union of routes below, attach route/rank evidence,
deduplicate, then apply an auditable final candidate policy.  That final set—not
an earlier broad union—is `candidate_pairs.tsv` and the input to the matcher.

1. **Exact high-precision routes:** `(country, rare normalised name token)`,
   `(country, postal/PIN token, name prefix)`, `(country, house number, rare
   street/name token)`, exact normalised name + address token intersection.
2. **Sparse name retrieval:** character 3--5 gram TF-IDF cosine top-K within
   country; a separate word/BM25 or rare-token route for long names.  Query both
   original-order and suffix-free normalised names.
3. **Sparse address retrieval:** character grams and word/token retrieval within
   country, with a small top-K.  This rescues a heavily abbreviated name when an
   address survives.
4. **Fallback routes:** carefully capped cross-country/unknown-country retrieval
   only when country is empty or contradictory; phonetic keys only after the
   training ablation proves benefit and with a maximum block size.  Do not let
   common terms (“services”, “trading”, “hospital”, etc.) create giant blocks.
5. **Adaptive budget:** the normal case receives a small union budget; allocate
   more candidates only when retrieval signals disagree, the name is short or
   common, the address is missing, or the S1 has multiple plausible strong
   candidates.  Do not cap to top-1: training shows true match counts above one.

Candidate-policy tuning should report `link recall`, `entity recall` (all true
links retained for an S1), total pairs, mean/median/p95/p99 candidates per S1,
maximum block, and candidate count by country and match multiplicity.  Measure
each route's marginal true-link recovery per added candidate.  Drop dominated
routes: ones adding candidate volume without recovering held-out links.

### Stage C — supervised pair reranker

Use LightGBM/XGBoost/CatBoost only after confirming license/version and baseline
performance; otherwise histogram gradient boosting is a viable baseline.  Trees
fit heterogeneous, non-linear string evidence and produce much cheaper
inference than a transformer cross-encoder.

Train positives from ground truth.  Negatives must come primarily from *the same
retrieval blocks* as positives: nearest name collisions, same-address different
name records, same-country candidates, common-name candidates, and high-scoring
false candidates from a previous model.  Random global nonmatches are nearly
free/easy and teach the wrong boundary.  Sample weights or a two-stage training
schedule can retain broad negatives without letting them dominate.

Build at least these features:

| family | examples |
|---|---|
| Name | char n-gram cosine; normalized/edit similarity; Jaro-Winkler; token Jaccard and weighted containment; token order; acronym; rare-token agreement; legal-suffix-only difference; length ratio |
| Address | char/word cosine; token Jaccard/containment; extracted number and postal equality/conflict; address length ratio; rare locality/street overlap; missingness |
| Cross-field | country equality/missing/conflict; name-address aggregate; exact-view flags; source pair (S1-S2 vs S1-S3) |
| Retrieval/ambiguity | every route flag, per-route rank/score, candidate count, nearest-vs-second-nearest score gap, frequency of shared name/address tokens |

Avoid treating any single common name or same address as conclusive.  Franchises,
multi-tenant buildings, shared corporate addresses, and branches are expected
hard negatives.

### Stage D — F0.5 decision policy

Calibrate scores on held-out S1 entities (isotonic or Platt only if it improves
reliability), but optimise decisions against the exact organiser macro metric,
not pairwise AUC, accuracy, or F1.  Search:

* a global high threshold;
* threshold by source pair and data-quality/ambiguity bucket, only if each has
  adequate validation support;
* margin requirements for ambiguous top candidates;
* high-confidence deterministic accepts and strong-conflict vetoes;
* an explicit empty-set/abstain outcome.

Do not infer that F0.5 makes a fixed probability 0.5 optimal.  The metric is
non-decomposable at the set level, macro-averages S1 entities, awards singleton
empties, and the expected cost of adding a pair depends on that S1's other
candidates.  Select the policy by end-to-end validation.  Permit several links
when their independently calibrated evidence clears policy; do not impose
one-to-one target assignment unless training ground truth demonstrates that
constraint.

## 5. Validation that does not lie

Split on S1 IDs before any model fitting or threshold selection, stratifying by
country, singleton status and match multiplicity.  Fit normalisation dictionaries
that are learned from labels, IDF/index statistics if practical, model,
calibration, and candidate-policy settings on training folds only.  Search the
held-out S1 fold against the complete eligible target corpus, then score the
organiser's per-S1 macro F0.5 exactly.

Random pairwise splits leak badly: the same S1/target representation, normalised
variants, easy duplicate pattern, or near-identical pairs can occur in both
train and validation.  They also overrepresent easy random negatives.  Such a
model can have impressive pairwise AUC yet make bad retrieval and false-merge
decisions on unseen businesses.  Use an untouched final S1 holdout after
iteration; use several folds or a large fixed development/holdout split if time
allows.

Report four separate layers:

1. candidate link recall and candidate efficiency;
2. reranker pair PR curve on candidates;
3. final macro F0.5, singleton accuracy, precision and recall;
4. error slices: France cannot be labelled in train, so use India/US quality
   slices plus test-only retrieval diagnostics without pretending test labels.

`candidate recall × reranker recall` is a useful upper-bound intuition for link
recall, but not a metric identity and says nothing directly about precision or
macro singleton behaviour.

## 6. Baseline ladder and stop/go gates

| stage | experiment | what it tests | advance when |
|---|---|---|---|
| 0 | exact multi-view name/address rules | data quality and easy-match ceiling | results are valid; retain only precise rules |
| 1 | capped fuzzy/character name + address retrieval | whether corruption is mostly lexical | candidate recall materially improves within budget |
| 2 | multi-route blocking with route diagnostics | recall per candidate spent | every kept route has marginal value |
| 3 | engineered features + GBDT | separation of hard retrieval negatives | macro F0.5 beats rules at same/lower budget |
| 4 | character TF-IDF top-K + reranker | sparse retrieval beyond exact blocks | dominates prior candidate-recall/size frontier |
| 5 | hybrid sparse plus dense ANN | whether embeddings recover sparse misses | held-out recovery beats added candidates/compute and no precision loss |
| 6 | neural reranker/contrastive bi-encoder | difficult semantic/transliteration cases | beats GBDT on the *same candidates*, including hard slices |

Stages 5--6 are conditional, not prizes for sophistication.  A cross-encoder is
only plausible as a reranker for a small fixed candidate set; it cannot score
the unblocked Cartesian product.  A bi-encoder must use hard negatives and ANN
recall evaluation, and its pretraining/license must be explicitly challenge-safe.

## 7. Ablations and red-team checklist

Run one change at a time against a frozen S1 validation split and candidate
budget.  Ablate: every normalised view; each blocking/retrieval route; name vs
address features; numeric/postal agreement and conflict; country treatment;
route rank; ambiguity features; hard-negative mining; decision margin; dynamic
thresholds; dense retrieval; neural reranking.  Record deltas in all four
validation layers above.

Review false positives and false negatives in these mandatory slices:

* common/short names; legal-suffix-only differences; acronyms and reordered
  names; transliterated or non-Latin text;
* missing, partial, duplicated, or contradictory addresses; same building and
  franchise/branch cases; house/postal number conflicts;
* high candidate-count S1s and S1s with 0, 1, and many true matches;
* country mismatch/empty values and France test retrieval coverage;
* candidates recovered by only one route; final accepted pairs without a clear
  retrieval rationale.

Any graph-based post-processing must be evaluated as an ablation with component
size/error monitoring.  Do not accept transitive closure merely because pair
scores are individually high.

## 8. Compute and reproducibility plan

Stream TSVs in chunks; use on-disk sparse matrices/inverted indexes or a
DuckDB/SQLite staging layer, and persist versioned normalisation/index/candidate
artifacts.  Separate S2 and S3 indexes, batch S1 queries, and cap pathological
blocks before materialising pair features.  Store candidate IDs plus route/rank
metadata once, then feature-score in chunks.  This avoids holding 10M targets,
all pairs, and all feature rows in Python memory.

Pin dependencies after a tested run, save split IDs/configuration/feature list,
and make the CLI regenerate both TSVs.  The final candidate writer must be the
same object passed to inference, sort/deduplicate IDs, and run the supplied
validator.  The official validator is necessary for format correctness but does
not validate candidate quality or compute F0.5.

## Final recommendation: what to implement first

1. Add a streaming EDA command and exact organiser-metric implementation.
2. Create an S1-level development/holdout split and a candidate-evaluation
   report including recall **and candidate budget distribution**.
3. Implement normalisation plus exact/rare-token and character-TF-IDF retrieval
   routes with strict, configurable budgets.
4. Materialise the final candidate set with route/rank metadata and establish
   the Pareto frontier before writing a sophisticated model.
5. Train the hard-negative GBDT reranker and tune the set-level F0.5 policy.
6. Only then test dense retrieval or a neural reranker on demonstrated sparse
   failure modes.

This sequence directly satisfies the revised ranking incentive: it makes the
smallest defensible final candidate set measurable and reproducible, rather
than hiding a huge candidate union behind a good matcher.

## Research coverage and limitations

Source lanes used: the supplied challenge resource and local archive; original
Fellegi--Sunter theory; peer-reviewed/academic entity-matching work; Splink's
current scalable-linkage documentation; and FAISS's official ANN documentation.
The research stopped after independent sources converged on the need for
blocking/retrieval plus pairwise decisions, and after comparing neural and graph
alternatives against the specific scale and F0.5 constraints.  Missing evidence
is intentionally visible: no model has been trained, no candidate recall has
been measured, no public implementation score is used as a benchmark, and the
organiser has not published a candidate-size scoring formula.
