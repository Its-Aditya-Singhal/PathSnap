# PathSnap: Phase 1 viva prep (models and training)

This is a study sheet for the team, built from our own code and results. Everything is on the 80 train trips and the 10 student_test trips.
Do not paste it into the report. The report is checked for AI-written text, so write those answers in your own words.
Tuning numbers below are final (from results/tuning/).

---

## A. The big picture

**Q1. What problem are we solving, in one line?**
Each trip has noisy GPS points and its own road network. For every GPS point we decide which road arc the car was on, then join those arcs into one connected route.

**Q2. Why not just snap each point to the nearest road?**
We measured it. Nearest-arc gets only **52.5% Top-1** on student_test (55.1% on train). Most roads are two-way: about 92% of arcs have a reverse arc at exactly the same distance. Distance alone can't tell the two directions apart, and GPS error (median 4.9 m, p95 17 m) often makes a parallel or crossing road the closest one.

**Q3. What is our overall approach?**
It has two layers.
1. **A classifier** scores each candidate arc for each GPS point: P(this candidate is the true arc). It uses 18 features: distance, heading agreement, neighbouring-point fit, and so on.
2. **An HMM with Viterbi decoding** picks the best *sequence* of arcs. Emission = how good each candidate looks at that point (from the classifier). Transition = how plausible it is to drive from one arc to the next on the road network.

The HMM is the base because the problem is sequential: a car can't teleport between roads. The classifiers plug in as the emission model.

**Q4. Why is it a binary classification task, not multi-class?**
Arc IDs are local to each trip's network, and there are thousands of arcs per network (median 19.5k on train). There are no fixed class labels across trips. So we turn it into "candidate vs. true arc": each (point, candidate) pair is one row with label 1 if it is the true arc, else 0. For each point, the highest-scoring candidate is the prediction.

---

## B. Data, candidates and labels

**Q5. What does the data look like?** (from `results/eda/eda_summary.json`)
- **Train:** 80 trips, 208,417 GPS points, about 1,500 points per trip, median sampling interval 1 s, median trip about 28 min and 14 km of route.
- **Gaps:** 22 of the 80 train trips have a gap over 60 s (one gap is 153,064 s).
- **Locations:** trips are spread worldwide, and each has its own road network.
- **student_test:** 10 trips, 19,160 points. Only 1 trip has a gap over 60 s.

**Q6. How are candidates generated?**
- We project every trip into a local metric frame: an azimuthal-equidistant projection centred on that trip, so all distances are in metres.
- We sample points every 10 m along each arc and build a KD-tree on them.
- For each GPS point we take up to **k = 8** arcs within **50 m**.
- **Candidate recall:** the true arc is in the candidate set for 96.1% of train points and 98.6% of student_test points. The remaining ~4% is a hard ceiling on Top-1 for any model.
- There are about 5.7 candidates per point, and 16.8% of candidate rows are positives (class imbalance roughly 1 : 5).

**Q7. Why 50 m / k = 8?**
- GPS error to the true arc is median 4.9 m and p95 17 m on train, so 50 m covers almost all real points while keeping the candidate set small.
- k = 8 keeps both directions of the nearby roads.
- We chose these values on train only.

**Q8. The `.route` file is a list of arcs, not one per GPS point. How did you get per-point labels?**
- We run the same HMM, but only route arcs are allowed as candidates.
- This assigns each point to a route arc in driving order, which handles U-turns, loops and two-way roads better than "nearest route arc".
- Points with no route arc within 150 m are marked unlabelled (-1) and left out of point-level metrics.
- The code is in `src/pathsnap/labels.py`. Labels are used only for training and for server-side scoring, never as a model input.

**Q9. Any weakness in the labels?**
Yes, and we state it openly. The labelling HMM uses a distance-based emission, so labels lean toward distance-based matching. That partly explains why the plain distance HMM is such a strong baseline (see Q30).

---

## C. Features (18 per point–candidate pair, `src/pathsnap/features.py`)

**Q10. What features did you use and why?**

| Group | Features | Why |
|---|---|---|
| Distance | `dist`, `rank`, `dist_minus_min`, `dist_ratio_min`, `n_cands` | Closeness, in absolute terms and relative to the other candidates |
| Direction | `heading_diff`, `heading_cos`, `win_heading_diff` | The main way to separate the two directions of a two-way road |
| Temporal window | `win_dist_mean`, `win_dist_max` (±3 points) | A true arc should also fit the neighbouring points; this smooths out single-point noise |
| Motion | `gps_speed`, `gps_disp`, `dt_prev` (log) | Heading is unreliable when the car is slow or stopped; gaps matter |
| Arc / graph | `arc_length`, `t_offset`, `has_reverse`, `out_deg_v`, `in_deg_u` | Road structure: two-way roads, junctions, position along the arc |

**Q11. Do the features ever use the reference route?**
No. Features use only the GPS track and the road network, so the same code runs unchanged on evaluator trips. The route is used only to build training labels.

**Q12. What scaling did you do?**
- Logistic regression, GNB, kNN and the MLP sit in a pipeline with a `StandardScaler`.
- In CV, each fold's scaler is fitted on that fold's training part only, so there is no leakage.
- Tree models (RF, LightGBM, XGBoost) don't need scaling.

---

## D. Why these models

**Q13. Which models did you train?**
Seven classifiers from six families, plus the HMM-only baseline:

| Model | Family |
|---|---|
| Logistic Regression | Linear |
| Gaussian Naive Bayes | Probabilistic |
| k-Nearest Neighbours | Instance-based |
| Random Forest | Ensemble (bagging) |
| LightGBM | Ensemble (boosting) |
| XGBoost | Ensemble (boosting) |
| MLP | Neural network |

**Q14. Why these specifically?**
- **Different families.** The guidelines ask for a spread of algorithm families, and each one makes a different assumption.
  - Linear: features combine additively.
  - Naive Bayes: features are independent given the class.
  - kNN: similar rows have similar labels.
  - Trees: thresholds and interactions.
  - MLP: smooth non-linear functions.

  Comparing them tells us *which kind* of model suits the data.
- **They must output probabilities.** The HMM needs P(candidate) as an emission, not just a yes/no. All seven give `predict_proba`. That is why we excluded models like a plain SVM, which needs extra calibration.
- **Tabular data with interactions.** Our features interact. For example, heading matters only when speed is high enough, and distance matters relative to the nearest candidate. Tree ensembles are known to work best on this kind of tabular data, so we included three (RF, LightGBM, XGBoost).
- **Simple baselines** (LogReg, GNB) show how much the non-linear models actually add.

**Q15. Why two boosting libraries?**
They differ in tree growth: LightGBM grows leaf-wise, XGBoost depth-wise with a histogram method. They also differ in speed. They ended up statistically tied, which is itself a useful finding.

---

## E. Training procedure (`scripts/train_models.py`)

**Q16. Walk us through the training procedure.**
1. Build the candidate table: about 1.08 M labelled rows from the 80 train trips (`scripts/build_dataset.py`).
2. **5-fold GroupKFold, grouped by trip.** All rows of a trip are in the same fold, so the model is always validated on trips it has never seen.
3. In each fold: fit on 4 folds, predict probabilities on the 5th, then record Macro F1, log loss, Top-1 and fit time.
4. **Refit** on all 80 train trips with the same hyperparameters.
5. Evaluate once on the 10 student_test trips: **classifier-only** (argmax per point) and **classifier + HMM/Viterbi**.
6. Save everything: `results/model_comparison.csv`, `cv_folds.csv`, per-trip results, confusion matrices, loss histories and fitted models.

Fixed seed (42) everywhere.

**Q17. Why GroupKFold and not normal KFold?**
Points within one trip are highly correlated: neighbouring points sit 1 s apart on the same road. Random KFold would put almost-identical rows in train and validation and give an inflated score. Grouping by trip simulates the real task, which is predicting a *new* trip.

**Q18. What loss / objective does each model optimise?**
- **Logistic regression, MLP, LightGBM, XGBoost:** binary cross-entropy (log loss).
- **Random forest:** Gini impurity at each split.
- **Gaussian NB:** fitted by maximum likelihood; there is no iterative loss.
- **kNN:** no training loss; it just stores the data.

**Q19. Which metrics did you use and why?**
- **Top-1 accuracy** (per GPS point): is the predicted arc the true arc? This is the main task metric.
- **Macro F1** (binary "is true arc", threshold 0.5, averaged over both classes): it treats the minority positive class (16.8%) fairly.
- **Log loss:** it measures probability quality. This matters because the HMM uses the probabilities directly.
- **Route F1 and continuity:** they measure whether the final route is right and connected.
- **Inference time** per point.

**Q20. How did you make sure student_test wasn't used for decisions?**
- Model choice, hyperparameters, search space and HMM settings are all chosen from train-only CV.
- student_test is evaluated *after* decisions are fixed, only for reporting.
- Evaluator trips are never seen.

---

## F. Per-model questions (the report asks these for EVERY model)

Numbers: CV = 5-fold GroupKFold on train (mean ± std). Test = student_test. Top-1 is "clf-only → +HMM".

### Logistic Regression (C = 1.0, StandardScaler, max_iter 2000)
- **Why suitable:** a fast, well-calibrated linear baseline that outputs probabilities directly.
- **Strengths:** fastest to train (1.3 s) and to predict; easy to interpret.
- **Limitations:** it can't model interactions such as "heading only matters when moving", and it has no thresholds.
- **Results:**
  - CV Macro F1 0.900 ± 0.017, CV log loss 0.160.
  - Test Macro F1 0.903, log loss 0.149, Top-1 0.853 → 0.900.
- **Fit:** it **underfits**. It is the second-worst on every metric and well below the tree models, while CV and test are close. That is a capacity problem, not overfitting.

### Gaussian Naive Bayes (var_smoothing 1e-9)
- **Why suitable:** a very fast probabilistic baseline.
- **Strengths:** trains in 0.14 s and needs little data.
- **Limitations:** it assumes the features are independent, but ours are strongly correlated (`dist`, `dist_minus_min`, `rank` and the window distances all measure closeness). It therefore counts the same evidence several times and becomes **overconfident**.
- **Results:**
  - CV Macro F1 0.836 ± 0.021, CV log loss **0.620**, the worst by far.
  - Test Macro F1 0.833, log loss 0.625, Top-1 0.835 → 0.893.
- **Fit:** it underfits, and its probabilities are poorly calibrated, as the high log loss shows.

### k-Nearest Neighbours (k = 25, distance-weighted, StandardScaler, trained on a 200k-row subsample)
- **Why suitable:** non-parametric; it makes no assumption about the shape of the decision boundary.
- **Strengths:** no training cost (fit 0.07 s).
- **Limitations:**
  - It has the slowest inference by far, **0.59 ms per point** (25–700× the others), because every prediction searches the stored rows.
  - Memory forced a 200k-row subsample.
  - All features count equally after scaling.
- **Results:**
  - CV Macro F1 0.906 ± 0.021, CV log loss 0.202.
  - Test Macro F1 0.913, log loss 0.180, Top-1 0.856 → 0.897.
- **Fit:** it sits between the linear and tree models. Its log loss is worse than LogReg's, because neighbour-vote probabilities are coarse.

### Random Forest (200 trees, min_samples_leaf 20, max_features √)
- **Why suitable:** handles feature interactions and thresholds; robust to outliers and unscaled features.
- **Strengths:** among the best scores; stable across folds (lowest CV std, 0.015).
- **Limitations:** the slowest to train (68 s per fold, 88 s on the full data); large model file.
- **Results:**
  - CV Macro F1 0.921 ± 0.015, CV log loss 0.115.
  - Test Macro F1 0.924, log loss 0.105, Top-1 0.869 → **0.912**, the best classifier+HMM result.
- **Fit:** no sign of overfitting. Test is at or slightly above CV, and min_samples_leaf 20 limits how deep the trees memorise.

### LightGBM (400 trees, lr 0.05, 63 leaves; subsample 0.8, colsample 0.8, λ 1.0)
- **Why suitable:** gradient boosting is the strongest family on tabular data. It optimises log loss directly, which matches what the HMM needs.
- **Strengths:**
  - The best CV Macro F1 (0.9223) and log loss tied with RF (0.1147).
  - About 6× faster to train than RF (11 s per fold).
  - It records real train/validation loss curves.
- **Limitations:** more hyperparameters, so it needs tuning; leaf-wise growth can overfit.
- **Results:**
  - CV Macro F1 0.922 ± 0.016, CV log loss 0.115.
  - Test Macro F1 0.925, log loss 0.104, Top-1 0.872 → 0.902.
- **Fit:** **mild overfitting at the default 400 trees.** In the fold-0 loss curve (`results/plots/loss_curves.png`):
  - Validation loss is lowest at iteration **93** (0.132) and ends at 0.142.
  - Training loss keeps falling to 0.069.

  That growing gap is the evidence. It is the main reason we tuned it (see section H).

### XGBoost (400 trees, lr 0.05, max_depth 6, hist method)
- **Why suitable:** same reasons as LightGBM, but it grows trees depth-wise.
- **Strengths:**
  - Best test Macro F1 and log loss by a hair (0.926 / 0.104).
  - Fast: 7 s per fold, and 0.004 ms per point to predict.
- **Limitations:** like LightGBM, it needs tuning.
- **Results:**
  - CV Macro F1 0.922 ± 0.016, CV log loss 0.115.
  - Test Macro F1 0.926, log loss 0.104, Top-1 0.873 → 0.902.
- **Fit:** the same mild overfitting pattern. Validation is best at iteration **106** (0.132) and ends at 0.139, while training ends at 0.078.

### MLP (64-32 hidden units, α 1e-4, Adam lr 1e-3, batch 1024, 15 epochs, StandardScaler)
- **Why suitable:** learns smooth non-linear combinations of the features; it is the neural-network family.
- **Strengths:** fast to predict; it gives real per-epoch loss curves.
- **Limitations:** sensitive to scaling and to the learning rate; on tabular data it is usually beaten by trees, and that happened here.
- **Results:**
  - CV Macro F1 0.914 ± 0.019, CV log loss 0.126.
  - Test Macro F1 0.920, log loss 0.116, Top-1 0.865 → 0.905.
- **Fit:** it **overfits early**. Validation loss is best at **epoch 3** (0.145) and ends at 0.147 after 15 epochs, while training loss keeps dropping (0.098). Early stopping around epoch 3–5 would help.

### HMM-only baseline (Gaussian distance emission σ = 5 m, β = 5 m)
- This is not a classifier; it is our base decoder with emission = −½(dist/σ)².
- **Results:** Top-1 0.525 (nearest-arc) → **0.924** with Viterbi. Route F1 0.986; continuity 0.894 → 0.999.

**Q21. Did any model overfit to the train trips overall?**
No. For every model, student_test Macro F1 is equal to or slightly above CV Macro F1. That means the models carry over to new trips. student_test is also a little easier: candidate recall is 98.6% vs 96.1%, and only 1 of its trips has a large gap. The overfitting we did see is *within training*, from too many boosting rounds or epochs, and it shows in the loss curves.

**Q22. Why don't RF, kNN, LogReg and GNB have loss curves?**
They don't train in epochs or iterations, so there is no real per-epoch history. We were told not to draw artificial curves, so the loss-curve plot covers only LightGBM, XGBoost and the MLP.

---

## G. Comparison and model choice

**Q23. Summary table (student_test unless marked CV).**

| Model | CV Macro F1 | CV log loss | Test Top-1 clf | Test Top-1 +HMM | Test Macro F1 | Test log loss | Route F1 +HMM |
|---|---|---|---|---|---|---|---|
| HMM baseline | — | — | 0.525 (nearest) | **0.924** | — | — | 0.986 |
| LogReg | 0.900 | 0.160 | 0.853 | 0.900 | 0.903 | 0.149 | 0.988 |
| GNB | 0.836 | 0.620 | 0.835 | 0.893 | 0.833 | 0.625 | 0.983 |
| kNN | 0.906 | 0.202 | 0.856 | 0.897 | 0.913 | 0.180 | 0.986 |
| RF | 0.921 | 0.115 | 0.869 | 0.912 | 0.924 | 0.105 | 0.989 |
| LightGBM | **0.922** | 0.115 | 0.872 | 0.902 | 0.925 | 0.104 | 0.989 |
| XGBoost | 0.922 | 0.115 | 0.873 | 0.902 | 0.926 | 0.104 | 0.989 |
| MLP | 0.914 | 0.126 | 0.865 | 0.905 | 0.920 | 0.116 | 0.989 |

**Q24. Which model did you pick for tuning, and why?**
**LightGBM**, on train CV evidence only:
- It has the highest CV Macro F1 (0.9223) and the joint-best CV log loss.
- It trains about 6× faster than RF, which matters because Bayesian tuning refits it many times.
- It produces loss curves that showed room to improve (the overfitting after about 93 trees).

RF, LightGBM and XGBoost are within 0.001 Macro F1 of each other. The fold-to-fold std (about 0.015) is much larger than that gap, so the top three are statistically tied, and speed and tunability broke the tie.

**Q25. What does the HMM add on top of a classifier?**
Large gains in every case:
- Top-1 rises 3–6 points (for example LightGBM 0.872 → 0.902).
- Route continuity rises from about 0.97 to 0.999.
- Route F1 rises from about 0.96 to 0.989.

A classifier alone jumps between roads from point to point. The HMM forces a drivable, connected path.

**Q26. Where does the HMM not help?**
- **True arc not among the candidates:** about 1.4% of test points, a ceiling nothing can fix.
- **Long GPS gaps (> 60 s):** the chain is split and each part is decoded on its own.
- **Two-way roads when the car is slow:** heading is unreliable at low speed, so both directions look equally good.

**Q27. Which errors remain?**
Mostly direction confusion on two-way roads. The two directed arcs are at the same distance, so the model must rely on heading and on network continuity. See `results/plots/confusion_matrices.png`.

---

## H. Bayesian hyperparameter optimisation (`scripts/tune.py`)

**Q28. How did you do it?**
- **Tool:** Optuna with the **TPE** sampler (a Bayesian method, seed 42) and a **MedianPruner**, which stops a trial early if its running fold loss is worse than the median of earlier trials.
- **Record:** every trial is saved in `results/tuning/optuna.db` and in `*_trials.csv`, with params, score, state and duration.
- **The untuned defaults are trial 0,** so tuned vs untuned is measured inside the same study.

**Stage 1: LightGBM hyperparameters**
- **Objective:** minimise mean **5-fold GroupKFold log loss** on the 80 train trips.
  - Why log loss and not F1: the HMM consumes the probabilities, so calibration matters more than a thresholded F1. Macro F1 is still recorded per trial.
- **Search space:**

| Hyperparameter | Range | Scale | Why |
|---|---|---|---|
| n_estimators | 150–900 (step 50) | linear | The default overfits after about 93 rounds at lr 0.05; a lower lr needs more rounds |
| learning_rate | 0.01–0.2 | log | Its effect is multiplicative |
| num_leaves | 15–255 | log | Model complexity (leaf-wise growth) |
| min_child_samples | 10–300 | log | Limits overfitting; we have about 870k rows per fold |
| subsample | 0.5–1.0 | linear | Row sampling for regularisation |
| colsample_bytree | 0.4–1.0 | linear | Feature sampling; we have 18 correlated features |
| reg_lambda | 0.001–20 | log | L2 regularisation |

- **Budget:** up to 30 trials or 3,000 s, whichever comes first.
- **Results:**
  - 30 trials, 10 pruned, about 43 min of compute.
  - Best CV log loss **0.1133**, against 0.1147 untuned.
  - Best params: 600 trees, learning rate 0.010, 145 leaves, min_child_samples 50, subsample 0.69, colsample 0.85, reg_lambda 1.02.
  - HMM studies: 50 trials each, about 2.5 min each. Baseline best: σ = 21.8, β = 30.6, train Top-1 0.856. Hybrid best: classifier weight 0.10, train Top-1 0.854.
- **Refit:** the best configuration is refitted on all 80 train trips. Its out-of-fold probabilities feed stage 2.

**Stage 2: HMM / emission parameters**
- **Objective:** maximise pooled Top-1 after Viterbi on the 80 train trips.
  - It uses **out-of-fold** classifier probabilities, so no train trip is scored by a model that was trained on it. Without this, the emission probabilities would be unrealistically confident.
- **Baseline study:** σ ∈ [2, 30] m (log), β ∈ [1, 50] m (log).
- **Hybrid study:**
  - emission = w·log P_clf − ½(d/σ)², with w ∈ [0.05, 5] (log)
  - the distance term on or off, σ ∈ [2, 60]
  - β ∈ [1, 50]
- **Trials:** 50 per study.

**Q29. Did the tuned model outperform the untuned one?**
**Final:**
- The tuned distance-only HMM was chosen: train Top-1 0.856, against 0.854 for the hybrid.
- On student_test:
  - Tuned HMM: Top-1 0.929 and Route F1 0.988.
  - Hybrid: Top-1 0.927.
  - Tuned LightGBM + default HMM: 0.916.
  - Untuned: 0.902.
- Tuning helped every system, and the hybrid wanted only a small classifier weight, which is consistent with the label bias.

To check this yourself, compare CV Macro F1, student_test Top-1, student_test Macro F1, log loss and inference time, from `results/tuning/final_comparison.csv`. The final system (hybrid vs tuned baseline) is chosen by the **train** objective only. The rule is in `results/tuning/selection.json`.

**Q30. Why was classifier + HMM worse than the plain distance HMM before tuning?** (0.90–0.91 vs 0.924)
Our explanation, which the tuning stage tests:
1. **Label bias.** Labels were made by a distance-based HMM, so a distance emission is naturally aligned with them.
2. **Double counting.** The classifier already uses neighbouring points (the window features), then the HMM adds sequence smoothing on top. With weight 1, its sharp probabilities can override the transition evidence.

This is why we tuned the emission weight w and allowed a mix of the classifier and the distance term.

---

## I. HMM and Viterbi (`src/pathsnap/hmm.py`)

**Q31. What are the hidden states?**
The candidate arcs of each GPS point, with up to 8 per point. Each state also stores where the point falls along that arc.

**Q32. How are emissions calculated?**
log P(observation | arc) =
- baseline: −½(d/σ)²
- with a classifier: w·log P̂(arc), where P̂ is the classifier probability normalised over that point's candidates
- or the hybrid of both, which Optuna tunes.

**Q33. How are transitions calculated?**
- The Newson & Krumm (2009) rule: log P = −|d_route − d_gps| / β.
  - d_route is the shortest-path network distance between the two positions (Dijkstra, cached).
  - d_gps is the straight-line distance between the two GPS points.
- Small backward jitter on the same arc is allowed (15 m).
- A transition is impossible (−∞) if it is unreachable or would need an unrealistic speed (over 60 m/s plus a 200 m slack).

**Q34. How are missing candidates and big gaps handled?**
- A point with no candidate within 50 m is skipped, and the chain continues from the previous point.
- A time gap over **60 s** starts a new chain, and each chain is decoded on its own.

**Q35. What if consecutive points suggest disconnected roads?**
If no candidate at time t is reachable from any candidate at t−1 (every transition is −∞), we break the chain and restart decoding from the emissions at t. Inside a chain, consecutive decoded arcs are joined by shortest-path arcs to build one connected route.

**Q36. Did HMM decoding improve the route?**
Yes, for every model. See Q25 and the per-trip numbers in `results/student_test_per_trip.csv`.

---

## J. Tricky questions to expect

**Q37. Isn't 0.92 Top-1 low?**
- About 1.4–3.9% of points have the true arc outside the candidate set, a hard ceiling.
- Most of the remaining errors are the opposite direction of the *same* road, which barely changes the drawn route. That is why route F1 is 0.99 while Top-1 is 0.92.

**Q38. How do you know you didn't tune on the test set?**
- Every choice comes from train-only CV or OOF-based train Top-1, and the selection rule is written to `selection.json` before student_test is evaluated.
- The prediction code loads trips with `with_route=False`, and an assertion blocks the route from being passed in.

**Q39. What would you do with more time?**
- Early stopping for the boosting models and the MLP, based on the loss curves.
- Labels that don't depend on a distance HMM.
- Better low-speed heading features to separate the two directions of a road.
- Calibrating the classifier probabilities before using them as emissions.

**Q40. Who did what / can every member explain the code?**
Each member should be able to walk through one stage end to end. Assign these among yourselves:
- `build_dataset.py`, plus labels and candidates
- `features.py`
- `train_models.py` and `models.py`
- `tune.py`
- `hmm.py`
- the API and UI
