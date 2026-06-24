# Concepts

FIG operates on binary (correct/incorrect) response data from **students** answering
**items**. An item is a question or task (e.g. a specific problem on a test) that multiple
students can respond to. Each observation is one student's response to one item.

## Baseline

The baseline is built from training data. For each item, FIG computes a **base rate**: the
fraction of correct responses in the training split. By default, these are stabilized via
empirical Bayes shrinkage (Beta(2, 2) prior) toward the global mean, which helps when
items have few observations:

\[
p_{\text{base}}^{(i)} = \frac{n_{\text{correct}}^{(i)} + \alpha}{n_{\text{total}}^{(i)} + \alpha + \beta}
\]

where \(n_{\text{correct}}^{(i)}\) and \(n_{\text{total}}^{(i)}\) are the number of
correct and total responses to item \(i\) in the training data, and \(\alpha\) and
\(\beta\) are the prior parameters.

You can control shrinkage via the `use_shrinkage`, `alpha`, and `beta` parameters:

- `use_shrinkage=True, alpha=2, beta=2` (default): mild regularization.
- `alpha=1, beta=1`: uniform prior (Laplace smoothing).
- `use_shrinkage=False`: raw item means, no regularization.

The **baseline entropy** for item \(i\) is the binary entropy of its base rate:

\[
H(p) = -\left[ p \log p + (1 - p) \log (1 - p) \right]
\]

Items near 50% correct have high baseline entropy (maximum uncertainty to reduce), while
very easy or very hard items have low baseline entropy.

Items that appear in the eval set but not in the training set fall back to the global
training mean as their baseline.

## FIG-V (Validation)

FIG-V requires ground truth labels on the eval split. It compares the model's binary
cross-entropy (BCE) to the baseline entropy:

\[
\text{BCE}(\hat{y}, y) = -\left[ y \log \hat{y} + (1 - y) \log (1 - \hat{y}) \right]
\]

\[
\text{FIG-V} = 1 - \frac{\sum_j \text{BCE}(\hat{y}_j, y_j)}{\sum_j H(p_{\text{base}}^{(i_j)})}
\]

where \(\hat{y}_j\) is the model's predicted probability for eval observation \(j\),
\(y_j\) is the ground truth, and \(i_j\) is the item ID.

The intuition: both numerator and denominator measure uncertainty, so FIG is the
**fraction of baseline uncertainty removed** by the model. It has the same
\(1 - \frac{\text{remaining}}{\text{total}}\) structure as reliability in psychometrics.

### Interpretation

| Value | Meaning |
|---|---|
| FIG-V = 1 | Perfect predictions (zero cross-entropy). |
| FIG-V = 0 | Model is no better than item base rates. |
| FIG-V < 0 | Model is worse than item base rates. |
| FIG-V > 1 | Likely data leakage or numerical issues. The library warns when this happens. |

FIG-V is differentiable and can be used as a training loss (see
[Getting Started](getting-started.md#using-fig-as-a-pytorch-loss)).

## FIG-C (Confidence)

FIG-C does not require ground truth. It replaces cross-entropy with the **entropy of the
model's own predictions**:

\[
\text{FIG-C} = 1 - \frac{\sum_j H(\hat{y}_j)}{\sum_j H(p_{\text{base}}^{(i_j)})}
\]

FIG-C measures how much more **confident** the model is relative to the baseline. If the
model outputs probabilities near 0 or 1, FIG-C is high. If it outputs probabilities near
0.5, FIG-C is low.

For well-calibrated models, FIG-C approximates FIG-V in expectation. This makes FIG-C
useful as a production metric where ground truth is unavailable -- analogous to the
Standard Error of Measurement (SEM) in psychometrics.

### Calibration dependence

FIG-C is based entirely on model confidence, not correctness. A miscalibrated model can
produce high FIG-C by being confidently wrong. FIG-V penalizes this via the cross-entropy
numerator.

## Pooled vs. Student-Weighted

Both metrics are computed in two ways:

**Student-weighted** (`fig_v` / `fig_c`): FIG is computed separately for each student,
then averaged. Each student gets equal weight regardless of how many items they answered.
This is the **primary metric**.

\[
\text{FIG-V}_{\text{student}} = \frac{1}{S} \sum_{s=1}^{S} \left( 1 - \frac{\sum_{j \in s} \text{BCE}(\hat{y}_j, y_j)}{\sum_{j \in s} H(p_{\text{base}}^{(i_j)})} \right)
\]

where \(S\) is the number of students and \(j \in s\) denotes the eval observations
belonging to student \(s\).

**Observation-weighted** (`fig_v_pooled` / `fig_c_pooled`): All observations are pooled.
Students with more items contribute more.

\[
\text{FIG-V}_{\text{pooled}} = 1 - \frac{\sum_j \text{BCE}(\hat{y}_j, y_j)}{\sum_j H(p_{\text{base}}^{(i_j)})}
\]

where the sums are over all eval observations. The two variants are equal when every
student answers the same number of items.
