#' Fractional Information Gain (FIG-V and FIG-C) - R Implementation
#'
#' This module provides functions to compute FIG metrics for evaluating
#' predictive models that account for item difficulty. The R implementation
#' mirrors the Python version in fig.py.
#'
#' - FIG-V (Validation): Requires ground truth, uses cross-entropy
#' - FIG-C (Confidence): No ground truth needed, uses entropy (model confidence)
#'
#' @author Tom DHF

#' Coerce an identifier vector to integers, erroring on non-integer values
#'
#' Item and student IDs are required to be integer-valued. This converts the
#' input to integers and raises a clear error if any value is missing,
#' non-numeric, or fractional, rather than silently truncating (as.integer())
#' or coercing to NA.
#'
#' @param x Identifier vector (integer, numeric, or character)
#' @param arg_name Name of the argument, used in the error message
#'
#' @return Integer vector
#'
#' @keywords internal
coerce_integer_ids <- function(x, arg_name) {
  if (is.factor(x)) {
    x <- as.character(x)
  }
  xi <- suppressWarnings(as.integer(x))
  if (any(is.na(xi)) || (is.numeric(x) && any(x != xi))) {
    stop(sprintf(
      paste0(
        "%s must contain only integer-valued IDs ",
        "(no NA, fractional, or non-numeric values)"
      ),
      arg_name
    ))
  }
  xi
}

#' Validate and Prepare Eval Inputs
#'
#' Validates and prepares the evaluation inputs shared by FIG-V and FIG-C.
#' Ground truth (\code{y_eval}) is optional: supply it for FIG-V, where it is
#' validated and returned; leave it NULL for FIG-C. The number of eval
#' observations is defined by \code{y_pred_eval}.
#'
#' @param y_pred_eval Numeric vector of predicted probabilities (0 to 1)
#' @param y_eval Numeric vector of eval ground truth (0=incorrect, 1=correct),
#'   or NULL (the default) for FIG-C
#' @param item_id_eval Integer vector of item IDs for eval
#' @param student_id_eval Integer vector of student IDs for eval
#' @param y_train Numeric vector of training ground truth (0=incorrect, 1=correct)
#' @param item_id_train Integer vector of item IDs for training
#' @param eps Small constant for clipping probabilities (default: 1e-12)
#'
#' @return A list of the validated, prepared vectors. Contains \code{y_eval}
#'   only when it was supplied.
#'
#' @keywords internal
validate_and_prepare_inputs <- function(y_pred_eval, y_eval = NULL, item_id_eval,
                                         student_id_eval, y_train, item_id_train,
                                         eps = 1e-12) {
  # Convert to numeric vectors / integer IDs.
  y_pred_eval <- as.numeric(y_pred_eval)
  item_id_eval <- coerce_integer_ids(item_id_eval, "item_id_eval")
  student_id_eval <- coerce_integer_ids(student_id_eval, "student_id_eval")
  y_train <- as.numeric(y_train)
  item_id_train <- coerce_integer_ids(item_id_train, "item_id_train")

  # The number of eval observations is defined by the predictions.
  n_eval <- length(y_pred_eval)
  n_train <- length(y_train)

  if (n_eval == 0) {
    stop("y_pred_eval must not be empty")
  }
  if (n_train == 0) {
    stop("y_train must not be empty")
  }

  if (length(item_id_eval) != n_eval) {
    stop(sprintf("Length mismatch: item_id_eval (%d) != y_pred_eval (%d)",
                 length(item_id_eval), n_eval))
  }
  if (length(student_id_eval) != n_eval) {
    stop(sprintf("Length mismatch: student_id_eval (%d) != y_pred_eval (%d)",
                 length(student_id_eval), n_eval))
  }
  if (length(item_id_train) != n_train) {
    stop(sprintf("Length mismatch: item_id_train (%d) != y_train (%d)",
                 length(item_id_train), n_train))
  }

  # Training labels must be binary.
  valid_values <- c(0, 1)
  unique_train <- unique(y_train)
  if (!all(unique_train %in% valid_values)) {
    invalid <- setdiff(unique_train, valid_values)
    stop(sprintf("y_train contains invalid values: %s. Must be in {0, 1}",
                 paste(invalid, collapse = ", ")))
  }

  # Check finiteness before the range comparison: comparing NaN with < or >
  # yields NA, so the range check would otherwise raise R's cryptic
  # "missing value where TRUE/FALSE needed" instead of a clear message.
  if (any(!is.finite(y_pred_eval))) {
    stop("y_pred_eval contains NaN or inf values")
  }
  if (any(y_pred_eval < 0) || any(y_pred_eval > 1)) {
    stop("y_pred_eval must be in [0, 1]")
  }

  # Clip probabilities.
  y_pred_eval <- pmax(pmin(y_pred_eval, 1 - eps), eps)

  result <- list(
    y_pred_eval = y_pred_eval,
    item_id_eval = item_id_eval,
    student_id_eval = student_id_eval,
    y_train = y_train,
    item_id_train = item_id_train
  )

  # FIG-V additionally supplies and validates ground-truth labels; FIG-C omits
  # them and this block is skipped.
  if (!is.null(y_eval)) {
    y_eval <- as.numeric(y_eval)
    if (length(y_eval) != n_eval) {
      stop(sprintf("Length mismatch: y_eval (%d) != y_pred_eval (%d)",
                   length(y_eval), n_eval))
    }
    unique_eval <- unique(y_eval)
    if (!all(unique_eval %in% valid_values)) {
      invalid <- setdiff(unique_eval, valid_values)
      stop(sprintf("y_eval contains invalid values: %s. Must be in {0, 1}",
                   paste(invalid, collapse = ", ")))
    }
    result$y_eval <- y_eval
  }

  result
}

#' Compute Per-Item Baseline
#'
#' Computes per-item baseline probability correct from training data with
#' optional Empirical Bayes shrinkage.
#'
#' @param y_train Numeric vector of binary training outcomes (0 or 1)
#' @param item_id_train Integer vector of item IDs for training data
#' @param use_shrinkage Logical, whether to apply EB shrinkage (default: TRUE)
#' @param alpha Prior pseudo-count for successes in EB shrinkage (default: 2.0)
#' @param beta Prior pseudo-count for failures in EB shrinkage (default: 2.0)
#'
#' @return A list with:
#'   - baseline_lookup: named numeric vector mapping item_id (as character) to
#'     its baseline probability
#'   - global_mean: overall mean correctness
#'
#' @keywords internal
compute_item_baseline <- function(y_train, item_id_train,
                                   use_shrinkage = TRUE,
                                   alpha = 2.0, beta = 2.0) {
  # Global mean correctness; used as the fallback for items unseen in training.
  global_mean <- mean(y_train)

  # Per-item statistics as named numeric vectors, indexed by item ID coerced to
  # a character string. tapply() names are the sorted unique training item IDs.
  item_sums <- tapply(y_train, item_id_train, sum)
  item_counts <- tapply(y_train, item_id_train, length)

  if (use_shrinkage) {
    # Empirical Bayes shrinkage: (successes + alpha) / (total + alpha + beta)
    baseline_lookup <- (item_sums + alpha) / (item_counts + alpha + beta)
  } else {
    # Simple per-item mean. Every item present here has at least one
    # observation, so the denominator is always positive.
    baseline_lookup <- item_sums / item_counts
  }

  list(
    baseline_lookup = baseline_lookup,
    global_mean = global_mean
  )
}

#' Map Item Baseline Probabilities
#'
#' Creates a vector of baseline probabilities for evaluation items.
#'
#' @param item_id_eval Integer vector of item IDs for evaluation data
#' @param baseline_probs Named numeric vector mapping item_id (as character) to
#'   its baseline probability
#' @param global_mean Fallback value for items not in lookup
#' @param eps Small constant for clipping (default: 1e-12)
#'
#' @return Numeric vector of baseline probabilities
#'
#' @keywords internal
map_item_baseline <- function(item_id_eval, baseline_probs, global_mean,
                               eps = 1e-12) {
  # Handle empty case
  if (length(item_id_eval) == 0) {
    return(numeric(0))
  }

  # Vectorized lookup; items absent from the baseline fall back to global_mean.
  baseline_prob_vector <- baseline_probs[as.character(item_id_eval)]
  baseline_prob_vector[is.na(baseline_prob_vector)] <- global_mean

  # Clip values
  baseline_prob_vector <- pmax(pmin(baseline_prob_vector, 1 - eps), eps)

  as.numeric(baseline_prob_vector)
}

#' Check Calibration
#'
#' Computes Expected Calibration Error (ECE) by binning predictions
#' and comparing average confidence to average accuracy in each bin.
#'
#' @param y_pred Numeric vector of predicted probabilities
#' @param y_true Numeric vector of true binary outcomes
#' @param n_bins Number of bins for binning predictions (default: 10)
#' @param warn_threshold Warn if ECE exceeds this value (default: 0.1)
#'
#' @return A list containing:
#'   - ece: Expected Calibration Error
#'   - bin_accuracies: Accuracy in each non-empty bin
#'   - bin_confidences: Average confidence in each non-empty bin
#'   - bin_counts: Number of samples in each non-empty bin
#'
#' @export
check_calibration <- function(y_pred, y_true, n_bins = 10, warn_threshold = 0.1) {
  # Bin predictions
  bin_edges <- seq(0, 1, length.out = n_bins + 1)
  # Use findInterval which is similar to np.digitize but 1-indexed
  # Adjust for R's 1-indexing and handle edge cases
  bin_indices <- findInterval(y_pred, bin_edges[-c(1, length(bin_edges))]) + 1

  ece <- 0.0
  bin_accs <- c()
  bin_confs <- c()
  bin_counts <- c()

  for (bin_idx in seq_len(n_bins)) {
    in_bin <- bin_indices == bin_idx
    if (sum(in_bin) > 0) {
      bin_acc <- mean(y_true[in_bin])
      bin_conf <- mean(y_pred[in_bin])
      bin_count <- sum(in_bin)

      ece <- ece + (bin_count / length(y_pred)) * abs(bin_acc - bin_conf)

      bin_accs <- c(bin_accs, bin_acc)
      bin_confs <- c(bin_confs, bin_conf)
      bin_counts <- c(bin_counts, bin_count)
    }
  }

  if (ece > warn_threshold) {
    warning(sprintf("ECE %.3f exceeds threshold %.3f", ece, warn_threshold))
  }

  list(
    ece = ece,
    bin_accuracies = bin_accs,
    bin_confidences = bin_confs,
    bin_counts = bin_counts
  )
}

#' Binary Entropy
#'
#' Computes binary entropy H(p) = -[p*log(p) + (1-p)*log(1-p)].
#'
#' @param p Numeric vector of probabilities (should already be clipped)
#'
#' @return Numeric vector of binary entropy values
#'
#' @keywords internal
binary_entropy <- function(p) {
  -(p * log(p) + (1 - p) * log(1 - p))
}


#' Aggregate per-observation entropies into pooled and per-student FIG
#'
#' Shared aggregation core used by both FIG-C and FIG-V. The two variants differ
#' only in how \code{numerator} is computed: FIG-C uses the binary entropy of
#' model predictions; FIG-V uses the binary cross-entropy of predictions against
#' ground truth.
#'
#' @param numerator Numeric vector of per-observation numerators (entropy for
#'   FIG-C, cross-entropy for FIG-V)
#' @param entropy_baseline Numeric vector of per-observation baseline entropies
#' @param student_id_eval Integer vector of student IDs for eval observations
#' @param metric_name Name used in warning messages, e.g. "FIG-C" or "FIG-V"
#'
#' @return A list with:
#'   - fig_pooled: observation-weighted FIG (scalar)
#'   - fig: student-weighted FIG (scalar)
#'   - fig_by_student: per-student FIG values (named numeric vector)
#'
#' @keywords internal
compute_fig <- function(numerator, entropy_baseline, student_id_eval,
                        metric_name) {
  # Observation-weighted (pooled)
  fig_pooled <- 1 - sum(numerator) / sum(entropy_baseline)

  # Student-weighted: per-student ratio of sums, then averaged across students.
  # tapply() returns a named vector indexed by sorted unique student IDs.
  student_numerator_sum <- tapply(numerator, student_id_eval, sum)
  student_baseline_sum <- tapply(entropy_baseline, student_id_eval, sum)
  fig_by_student <- 1 - (student_numerator_sum / student_baseline_sum)
  fig <- mean(fig_by_student)

  # Sanity checks
  if (fig_pooled > 1 + 1e-6) {
    warning(sprintf(
      "%s pooled %.3f > 1; possible leakage or model issues.",
      metric_name, fig_pooled
    ))
  }
  if (fig > 1 + 1e-6) {
    warning(sprintf(
      "%s %.3f > 1; possible leakage or model issues.",
      metric_name, fig
    ))
  }

  list(
    fig_pooled = fig_pooled,
    fig = fig,
    fig_by_student = fig_by_student
  )
}


#' Fractional Information Gain for Confidence (FIG-C)
#'
#' Computes FIG-C, which measures the reduction in uncertainty about student
#' responses based on model confidence, without requiring ground truth labels.
#' This makes it suitable for production use where held-out data is unavailable,
#' analogous to the Standard Error of Measurement (SEM) in psychometrics.
#'
#' FIG-C = 1 - H_M(Y|X) / H_D(Y)
#'
#' where:
#'   H_D(Y) = sum of binary entropies using item base rates (prior uncertainty)
#'   H_M(Y|X) = sum of binary entropies using model predictions (remaining uncertainty)
#'
#' @param y_pred_eval Numeric vector of predicted probabilities (0 to 1)
#' @param item_id_eval Integer vector of item identifiers for eval observations
#' @param student_id_eval Integer vector of student identifiers for eval observations
#' @param y_train Numeric vector of ground truth on train split (0=incorrect, 1=correct)
#' @param item_id_train Integer vector of item identifiers for train observations
#' @param eps Small constant for clipping probabilities (default: 1e-12)
#' @param use_shrinkage Logical, if TRUE apply EB shrinkage to item baselines (default: TRUE)
#' @param alpha Prior pseudo-count for successes in EB shrinkage (default: 2.0)
#' @param beta Prior pseudo-count for failures in EB shrinkage (default: 2.0)
#'
#' @return A list containing:
#'   - fig_c_pooled: Observation-weighted FIG-C (scalar)
#'   - fig_c: Student-weighted FIG-C (scalar)
#'   - fig_c_by_student: Per-student FIG-C values (named numeric vector)
#'   - student_ids: Unique student IDs in sorted order (integer vector)
#'
#' @details
#' FIG-C is based purely on model confidence (entropy), not accuracy.
#' - FIG-C = 0 when model predictions match item base rates (no information gain)
#' - FIG-C = 1 when model is perfectly confident (entropy = 0)
#' - FIG-C can be negative if model adds uncertainty beyond the baseline
#' - For well-calibrated models, FIG-C approximates FIG-V in expectation
#' - Items in eval that do not appear in train use the global training mean as baseline
#'
#' @examples
#' # Simple example: model confident about its predictions
#' y_pred <- c(0.95, 0.05, 0.9, 0.1, 0.8)
#' items <- c(1, 1, 2, 2, 1)
#' students <- c(1, 1, 2, 2, 3)
#' y_train <- c(1, 1, 0, 1, 0)
#' items_train <- c(1, 1, 2, 2, 2)
#'
#' results <- fractional_information_gain_confidence(
#'   y_pred_eval = y_pred,
#'   item_id_eval = items,
#'   student_id_eval = students,
#'   y_train = y_train,
#'   item_id_train = items_train
#' )
#' cat(sprintf("FIG-C: %.3f\n", results$fig_c))
#'
#' @export
fractional_information_gain_confidence <- function(
    y_pred_eval,
    item_id_eval,
    student_id_eval,
    y_train,
    item_id_train,
    eps = 1e-12,
    use_shrinkage = TRUE,
    alpha = 2.0,
    beta = 2.0
) {
  # Validate and prepare inputs (no y_eval: FIG-C needs no ground truth)
  validated <- validate_and_prepare_inputs(
    y_pred_eval = y_pred_eval,
    item_id_eval = item_id_eval,
    student_id_eval = student_id_eval,
    y_train = y_train,
    item_id_train = item_id_train,
    eps = eps
  )

  y_pred_eval <- validated$y_pred_eval
  item_id_eval <- validated$item_id_eval
  student_id_eval <- validated$student_id_eval
  y_train <- validated$y_train
  item_id_train <- validated$item_id_train

  # Build per-item baseline from training data
  baseline_result <- compute_item_baseline(
    y_train, item_id_train, use_shrinkage, alpha, beta
  )
  baseline_lookup <- baseline_result$baseline_lookup
  global_mean_train <- baseline_result$global_mean

  # Map baseline to eval observations
  baseline_probs <- map_item_baseline(
    item_id_eval, baseline_lookup, global_mean_train, eps = eps
  )

  # Compute binary entropies
  # H_D(Y): Prior uncertainty based on item base rates
  entropy_baseline <- binary_entropy(baseline_probs)

  # H_M(Y|X): Remaining uncertainty based on model predictions
  entropy_model <- binary_entropy(y_pred_eval)

  fig_result <- compute_fig(
    numerator = entropy_model,
    entropy_baseline = entropy_baseline,
    student_id_eval = student_id_eval,
    metric_name = "FIG-C"
  )

  list(
    fig_c_pooled = fig_result$fig_pooled,
    fig_c = fig_result$fig,
    fig_c_by_student = fig_result$fig_by_student,
    student_ids = as.integer(names(fig_result$fig_by_student))
  )
}


#' Fractional Information Gain for Validation (FIG-V)
#'
#' Computes FIG-V on an evaluation split. FIG-V measures how much better a model
#' is than a per-item baseline:
#'
#' FIG-V = 1 - E[BCE_model] / E[H_baseline_train]
#'
#' where the denominator is the entropy of train-derived per-item base rates
#' (optionally Empirical-Bayes-shrunk) for the items in Y (the eval items).
#'
#' @param y_pred_eval Numeric vector of predicted probabilities on eval split (0 to 1)
#' @param y_eval Numeric vector of ground truth on eval split (0=incorrect, 1=correct)
#' @param item_id_eval Integer vector of item identifiers for eval observations
#' @param student_id_eval Integer vector of student identifiers for eval observations.
#'   Required for student-weighted averaging.
#' @param y_train Numeric vector of ground truth on train split (0=incorrect, 1=correct)
#' @param item_id_train Integer vector of item identifiers for train observations
#' @param eps Small constant for clipping probabilities (default: 1e-12)
#' @param use_shrinkage Logical, if TRUE apply EB shrinkage to item baselines (default: TRUE)
#' @param alpha Prior pseudo-count for successes in EB shrinkage (default: 2.0)
#' @param beta Prior pseudo-count for failures in EB shrinkage (default: 2.0)
#' @param calibration Logical, if TRUE attach calibration metrics (default: FALSE)
#' @param calib_n_bins Number of bins for ECE computation (default: 10)
#' @param calib_warn_threshold Warn if ECE exceeds this value (default: 0.1)
#'
#' @return A list containing:
#'   - fig_v_pooled: Observation-weighted FIG-V (scalar)
#'   - fig_v: Student-weighted FIG-V (scalar)
#'   - fig_v_by_student: Per-student FIG-V values (named numeric vector)
#'   - student_ids: Unique student IDs in sorted order (integer vector)
#'   - calibration: Calibration metrics if calibration=TRUE, otherwise NULL
#'
#' @details
#' FIG-V uses cross-entropy against ground truth, so it rewards accurate
#' predictions, not just confident ones.
#' - FIG-V = 0 when model predictions match item base rates (no information gain)
#' - FIG-V = 1 when model predictions are perfect (zero cross-entropy)
#' - FIG-V can be negative if the model is worse than the baseline
#' - FIG-V > 1 usually indicates data leakage; a warning is emitted when this happens
#' - Items in eval that do not appear in train use the global training mean as baseline
#'
#' @examples
#' # Simple example with 3 students, 2 items
#' y_pred <- c(0.8, 0.6, 0.9, 0.5, 0.7)
#' y_eval <- c(1, 0, 1, 0, 1)
#' items <- c(1, 1, 2, 2, 1)
#' students <- c(1, 1, 2, 2, 3)
#' y_train <- c(1, 1, 0, 1, 0)
#' items_train <- c(1, 1, 2, 2, 2)
#'
#' results <- fractional_information_gain_validation(
#'   y_pred_eval = y_pred,
#'   y_eval = y_eval,
#'   item_id_eval = items,
#'   student_id_eval = students,
#'   y_train = y_train,
#'   item_id_train = items_train
#' )
#' cat(sprintf("FIG-V: %.3f\n", results$fig_v))
#'
#' @export
fractional_information_gain_validation <- function(
    y_pred_eval,
    y_eval,
    item_id_eval,
    student_id_eval,
    y_train,
    item_id_train,
    eps = 1e-12,
    use_shrinkage = TRUE,
    alpha = 2.0,
    beta = 2.0,
    calibration = FALSE,
    calib_n_bins = 10,
    calib_warn_threshold = 0.1
) {
  # Validate and prepare inputs (y_eval supplied: FIG-V uses ground truth)
  validated <- validate_and_prepare_inputs(
    y_pred_eval = y_pred_eval,
    y_eval = y_eval,
    item_id_eval = item_id_eval,
    student_id_eval = student_id_eval,
    y_train = y_train,
    item_id_train = item_id_train,
    eps = eps
  )

  y_pred_eval <- validated$y_pred_eval
  y_eval <- validated$y_eval
  item_id_eval <- validated$item_id_eval
  student_id_eval <- validated$student_id_eval
  y_train <- validated$y_train
  item_id_train <- validated$item_id_train

  # Build per-item baseline from training data
  baseline_result <- compute_item_baseline(
    y_train, item_id_train, use_shrinkage, alpha, beta
  )
  baseline_lookup <- baseline_result$baseline_lookup
  global_mean_train <- baseline_result$global_mean

  # Compute cross-entropy for model predictions (numerator)
  # BCE = -[y*log(p) + (1-y)*log(1-p)]
  ce_model <- -(
    y_eval * log(y_pred_eval) +
      (1 - y_eval) * log(1 - y_pred_eval)
  )

  # Denominator: entropy of baseline (prior uncertainty) for Y items
  # Computed from training base rates with optional shrinkage
  baseline_probs_eval <- map_item_baseline(
    item_id_eval, baseline_lookup, global_mean_train, eps = eps
  )
  entropy_baseline <- binary_entropy(baseline_probs_eval)

  fig_result <- compute_fig(
    numerator = ce_model,
    entropy_baseline = entropy_baseline,
    student_id_eval = student_id_eval,
    metric_name = "FIG-V"
  )

  # Assemble result. calibration is always present (NULL unless requested) to
  # mirror the Python implementation's return shape.
  result <- list(
    fig_v_pooled = fig_result$fig_pooled,
    fig_v = fig_result$fig,
    fig_v_by_student = fig_result$fig_by_student,
    student_ids = as.integer(names(fig_result$fig_by_student)),
    calibration = NULL
  )

  # Optional calibration
  if (calibration) {
    result$calibration <- check_calibration(
      y_pred_eval, y_eval,
      calib_n_bins, calib_warn_threshold
    )
  }

  result
}
