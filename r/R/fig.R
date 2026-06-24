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
#' @version 1.1.0

# Helper function to check if a value is in a set
`%in_set%` <- function(x, set) {
  all(x %in% set)
}

#' Validate and Prepare Inputs
#'
#' Validates inputs and prepares data for FIG-V computation.
#'
#' @param y_pred_eval Numeric vector of predicted probabilities (0 to 1)
#' @param y_eval Numeric vector of ground truth (0=incorrect, 1=correct)
#' @param item_id_eval Integer vector of item IDs for eval
#' @param student_id_eval Integer vector of student IDs for eval
#' @param y_train Numeric vector of training ground truth (0=incorrect, 1=correct)
#' @param item_id_train Integer vector of item IDs for training
#' @param eps Small constant for clipping probabilities (default: 1e-12)
#'
#' @return A list containing validated and prepared vectors
#'
#' @keywords internal
validate_and_prepare_inputs <- function(y_pred_eval, y_eval, item_id_eval,
                                         student_id_eval, y_train, item_id_train,
                                         eps = 1e-12) {
  # Convert to numeric vectors and flatten
  y_pred_eval <- as.numeric(y_pred_eval)
  y_eval <- as.numeric(y_eval)
  item_id_eval <- as.integer(item_id_eval)
  student_id_eval <- as.integer(student_id_eval)
  y_train <- as.numeric(y_train)
  item_id_train <- as.integer(item_id_train)

  # Length checks
  n_eval <- length(y_eval)
  n_train <- length(y_train)

  if (length(y_pred_eval) != n_eval) {
    stop(sprintf("Length mismatch: y_pred_eval (%d) != y_eval (%d)",
                 length(y_pred_eval), n_eval))
  }
  if (length(item_id_eval) != n_eval) {
    stop(sprintf("Length mismatch: item_id_eval (%d) != y_eval (%d)",
                 length(item_id_eval), n_eval))
  }
  if (length(student_id_eval) != n_eval) {
    stop(sprintf("Length mismatch: student_id_eval (%d) != y_eval (%d)",
                 length(student_id_eval), n_eval))
  }
  if (length(item_id_train) != n_train) {
    stop(sprintf("Length mismatch: item_id_train (%d) != y_train (%d)",
                 length(item_id_train), n_train))
  }

  # Value checks
  valid_values <- c(0, 1)
  unique_eval <- unique(y_eval)
  unique_train <- unique(y_train)

  if (!all(unique_eval %in% valid_values)) {
    invalid <- setdiff(unique_eval, valid_values)
    stop(sprintf("y_eval contains invalid values: %s. Must be in {0, 1}",
                 paste(invalid, collapse = ", ")))
  }
  if (!all(unique_train %in% valid_values)) {
    invalid <- setdiff(unique_train, valid_values)
    stop(sprintf("y_train contains invalid values: %s. Must be in {0, 1}",
                 paste(invalid, collapse = ", ")))
  }

  if (any(y_pred_eval < 0) || any(y_pred_eval > 1)) {
    stop("y_pred_eval must be in [0, 1]")
  }
  if (any(!is.finite(y_pred_eval))) {
    stop("y_pred_eval contains NaN or inf values")
  }

  # Clip probabilities
  y_pred_eval <- pmax(pmin(y_pred_eval, 1 - eps), eps)

  list(
    y_pred_eval = y_pred_eval,
    y_eval = y_eval,
    item_id_eval = item_id_eval,
    student_id_eval = student_id_eval,
    y_train = y_train,
    item_id_train = item_id_train
  )
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
#'   - baseline_lookup: named list mapping item_id to baseline probability
#'   - global_mean: overall mean correctness
#'
#' @keywords internal
compute_item_baseline <- function(y_train, item_id_train,
                                   use_shrinkage = TRUE,
                                   alpha = 2.0, beta = 2.0) {
  # Compute global mean
  global_mean <- mean(y_train)

  # Get unique items
  unique_items <- unique(item_id_train)

  # Compute per-item statistics using tapply for efficiency
  item_sums <- tapply(y_train, item_id_train, sum)
  item_counts <- tapply(y_train, item_id_train, length)

  baseline_lookup <- list()
  for (item in unique_items) {
    item_str <- as.character(item)
    n_successes <- item_sums[item_str]
    n_total <- item_counts[item_str]

    if (use_shrinkage) {
      # Empirical Bayes shrinkage: (successes + alpha) / (total + alpha + beta)
      baseline_lookup[[item_str]] <- (n_successes + alpha) / (n_total + alpha + beta)
    } else {
      # Simple mean
      if (n_total > 0) {
        baseline_lookup[[item_str]] <- n_successes / n_total
      } else {
        baseline_lookup[[item_str]] <- global_mean
      }
    }
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
#' @param baseline_probs Named list mapping item_id to baseline probability
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

  # Vectorized lookup with fallback to global mean
  baseline_prob_vector <- sapply(item_id_eval, function(item) {
    item_str <- as.character(item)
    if (item_str %in% names(baseline_probs)) {
      baseline_probs[[item_str]]
    } else {
      global_mean
    }
  })

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

  for (bin_idx in 1:n_bins) {
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


#' Validate and Prepare Inputs for FIG-C
#'
#' Validates inputs and prepares data for FIG-C computation.
#' Similar to validate_and_prepare_inputs but doesn't require y_eval.
#'
#' @param y_pred_eval Numeric vector of predicted probabilities (0 to 1)
#' @param item_id_eval Integer vector of item IDs for eval
#' @param student_id_eval Integer vector of student IDs for eval
#' @param y_train Numeric vector of training ground truth (0=incorrect, 1=correct)
#' @param item_id_train Integer vector of item IDs for training
#' @param eps Small constant for clipping probabilities (default: 1e-12)
#'
#' @return A list containing validated and prepared vectors
#'
#' @keywords internal
validate_and_prepare_inputs_confidence <- function(y_pred_eval, item_id_eval,
                                                    student_id_eval, y_train,
                                                    item_id_train, eps = 1e-12) {
  # Convert to appropriate types
  y_pred_eval <- as.numeric(y_pred_eval)
  item_id_eval <- as.integer(item_id_eval)
  student_id_eval <- as.integer(student_id_eval)
  y_train <- as.numeric(y_train)
  item_id_train <- as.integer(item_id_train)

  # Length checks
  n_eval <- length(y_pred_eval)
  n_train <- length(y_train)

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

  # Value checks for training data
  valid_values <- c(0, 1)
  unique_train <- unique(y_train)

  if (!all(unique_train %in% valid_values)) {
    invalid <- setdiff(unique_train, valid_values)
    stop(sprintf("y_train contains invalid values: %s. Must be in {0, 1}",
                 paste(invalid, collapse = ", ")))
  }

  # Value checks for predictions
  if (any(y_pred_eval < 0) || any(y_pred_eval > 1)) {
    stop("y_pred_eval must be in [0, 1]")
  }
  if (any(!is.finite(y_pred_eval))) {
    stop("y_pred_eval contains NaN or inf values")
  }

  # Clip probabilities
  y_pred_eval <- pmax(pmin(y_pred_eval, 1 - eps), eps)

  list(
    y_pred_eval = y_pred_eval,
    item_id_eval = item_id_eval,
    student_id_eval = student_id_eval,
    y_train = y_train,
    item_id_train = item_id_train
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
#'   - student_ids: Unique student IDs (vector)
#'
#' @details
#' FIG-C is based purely on model confidence (entropy), not accuracy.
#' - FIG-C = 0 when model predictions match item base rates (no information gain)
#' - FIG-C = 1 when model is perfectly confident (entropy = 0)
#' - FIG-C can be negative if model adds uncertainty beyond the baseline
#' - For well-calibrated models, FIG-C ≈ FIG-V in expectation
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
  # Validate and prepare inputs
  validated <- validate_and_prepare_inputs_confidence(
    y_pred_eval, item_id_eval, student_id_eval,
    y_train, item_id_train, eps = eps
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

  # Observation-weighted FIG-C
  fig_c_pooled <- 1 - sum(entropy_model) / sum(entropy_baseline)

  # Student-weighted FIG-C
  student_baseline_sum <- tapply(entropy_baseline, student_id_eval, sum)
  student_model_sum <- tapply(entropy_model, student_id_eval, sum)

  # Compute per-student FIG-C
  fig_c_by_student <- 1 - (student_model_sum / student_baseline_sum)
  fig_c <- mean(fig_c_by_student)

  # Sanity checks
  if (fig_c_pooled > 1 + 1e-6) {
    warning(sprintf(
      "FIG-C pooled %.3f > 1; model predictions may be pathological.",
      fig_c_pooled
    ))
  }
  if (fig_c > 1 + 1e-6) {
    warning(sprintf(
      "FIG-C %.3f > 1; model predictions may be pathological.",
      fig_c
    ))
  }

  list(
    fig_c_pooled = fig_c_pooled,
    fig_c = fig_c,
    fig_c_by_student = fig_c_by_student,
    student_ids = as.integer(names(fig_c_by_student))
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
#'   - calibration: (optional) Calibration metrics if calibration=TRUE
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
  # Validate and prepare inputs
  validated <- validate_and_prepare_inputs(
    y_pred_eval, y_eval, item_id_eval, student_id_eval,
    y_train, item_id_train, eps = eps
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
  p_hat <- y_pred_eval

  ce_model <- -(
    y_eval * log(p_hat) +
      (1 - y_eval) * log(1 - p_hat)
  )

  # Denominator: entropy of baseline (prior uncertainty) for Y items
  # Computed from training base rates with optional shrinkage
  baseline_probs_eval <- map_item_baseline(
    item_id_eval, baseline_lookup, global_mean_train, eps = eps
  )
  entropy_baseline <- binary_entropy(baseline_probs_eval)

  # Observation-weighted FIG-V
  fig_v_interaction <- 1 - sum(ce_model) / sum(entropy_baseline)

  # Student-weighted FIG-V
  # Compute per-student sums using tapply
  unique_students <- unique(student_id_eval)
  student_model_sum <- tapply(ce_model, student_id_eval, sum)
  student_baseline_sum <- tapply(entropy_baseline, student_id_eval, sum)

  # Compute per-student FIG-V: ratio of sums per student
  fig_v_by_student <- 1 - (student_model_sum / student_baseline_sum)
  fig_v <- mean(fig_v_by_student)

  # Sanity checks
  if (fig_v_interaction > 1 + 1e-6) {
    warning(sprintf(
      "FIG-V interaction %.3f > 1; possible leakage or model issues.",
      fig_v_interaction
    ))
  }
  if (fig_v > 1 + 1e-6) {
    warning(sprintf(
      "FIG-V %.3f > 1; possible leakage or model issues.",
      fig_v
    ))
  }

  # Assemble result
  result <- list(
    fig_v_pooled = fig_v_interaction,
    fig_v = fig_v
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
