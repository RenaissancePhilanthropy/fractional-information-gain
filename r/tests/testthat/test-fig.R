#' Tests for fig.R - Fractional Information Gain for Validation
#'
#' Run with: testthat::test_dir("r/tests/testthat/")

library(testthat)

# ==============================================================================
# Tests for validate_and_prepare_inputs
# ==============================================================================

test_that("validate_and_prepare_inputs works with valid inputs", {
  y_pred <- c(0.8, 0.6, 0.7)
  y_eval <- c(1, 0, 1)
  item_id_eval <- c(1, 2, 1)
  student_id_eval <- c(1, 1, 2)
  y_train <- c(1, 0, 1, 0)
  item_id_train <- c(1, 1, 2, 2)

  result <- validate_and_prepare_inputs(
    y_pred, y_eval, item_id_eval, student_id_eval, y_train, item_id_train
  )

  expect_length(result, 6)
  expect_type(result$y_pred_eval, "double")
  expect_type(result$y_eval, "double")
  expect_type(result$item_id_eval, "integer")
  expect_type(result$student_id_eval, "integer")
})

test_that("validate_and_prepare_inputs clips probabilities", {
  eps <- 1e-12
  y_pred <- c(0.0, 1.0, 0.5)
  y_eval <- c(1, 0, 1)
  item_id_eval <- c(1, 2, 1)
  student_id_eval <- c(1, 1, 2)
  y_train <- c(1, 0)
  item_id_train <- c(1, 2)

  result <- validate_and_prepare_inputs(
    y_pred, y_eval, item_id_eval, student_id_eval, y_train, item_id_train, eps = eps
  )

  expect_gte(min(result$y_pred_eval), eps)
  expect_lte(max(result$y_pred_eval), 1 - eps)
})

test_that("validate_and_prepare_inputs errors on length mismatch", {
  y_pred <- c(0.8, 0.6)
  y_eval <- c(1, 0, 1)
  item_id_eval <- c(1, 2, 1)
  student_id_eval <- c(1, 1, 2)
  y_train <- c(1, 0)
  item_id_train <- c(1, 2)

  expect_error(
    validate_and_prepare_inputs(
      y_pred, y_eval, item_id_eval, student_id_eval, y_train, item_id_train
    ),
    "Length mismatch"
  )
})

test_that("validate_and_prepare_inputs errors on invalid y_eval values", {
  y_pred <- c(0.8, 0.6, 0.7)
  y_eval <- c(1, 2, 1)  # 2 is invalid
  item_id_eval <- c(1, 2, 1)
  student_id_eval <- c(1, 1, 2)
  y_train <- c(1, 0)
  item_id_train <- c(1, 2)

  expect_error(
    validate_and_prepare_inputs(
      y_pred, y_eval, item_id_eval, student_id_eval, y_train, item_id_train
    ),
    "invalid values"
  )
})

test_that("validate_and_prepare_inputs errors on invalid y_train values", {
  y_pred <- c(0.8, 0.6, 0.7)
  y_eval <- c(1, 0, 1)
  item_id_eval <- c(1, 2, 1)
  student_id_eval <- c(1, 1, 2)
  y_train <- c(1, -1)  # -1 is invalid
  item_id_train <- c(1, 2)

  expect_error(
    validate_and_prepare_inputs(
      y_pred, y_eval, item_id_eval, student_id_eval, y_train, item_id_train
    ),
    "invalid values"
  )
})

test_that("validate_and_prepare_inputs errors on out-of-range predictions", {
  y_pred <- c(0.8, 1.5, 0.7)
  y_eval <- c(1, 0, 1)
  item_id_eval <- c(1, 2, 1)
  student_id_eval <- c(1, 1, 2)
  y_train <- c(1, 0)
  item_id_train <- c(1, 2)

  expect_error(
    validate_and_prepare_inputs(
      y_pred, y_eval, item_id_eval, student_id_eval, y_train, item_id_train
    ),
    "must be in"
  )
})

test_that("non-finite predictions give a clear NaN/inf error (FIG-V validator)", {
  # Regression guard: the finiteness check must run before the range check, so
  # NaN does not trigger R's cryptic "missing value where TRUE/FALSE needed"
  # (from comparing NaN with <), and Inf is reported as non-finite rather than
  # as out-of-range.
  base <- list(
    y_eval = c(1, 0, 1), item_id_eval = c(1, 2, 1),
    student_id_eval = c(1, 1, 2), y_train = c(1, 0), item_id_train = c(1, 2)
  )
  for (bad in c(NaN, Inf, -Inf)) {
    expect_error(
      do.call(
        validate_and_prepare_inputs,
        c(list(y_pred_eval = c(0.8, bad, 0.7)), base)
      ),
      "NaN or inf"
    )
  }
})

test_that("non-finite predictions give a clear NaN/inf error (FIG-C)", {
  for (bad in c(NaN, Inf, -Inf)) {
    expect_error(
      fractional_information_gain_confidence(
        y_pred_eval = c(0.8, bad, 0.7),
        item_id_eval = c(1, 2, 1),
        student_id_eval = c(1, 1, 2),
        y_train = c(1, 0),
        item_id_train = c(1, 2)
      ),
      "NaN or inf"
    )
  }
})

# ==============================================================================
# Tests for compute_item_baseline
# ==============================================================================

test_that("compute_item_baseline works without shrinkage", {
  y_train <- c(1, 1, 0, 1, 0, 0)
  item_id_train <- c(1, 1, 1, 2, 2, 2)

  result <- compute_item_baseline(y_train, item_id_train, use_shrinkage = FALSE)

  # Item 1: 2/3 correct, Item 2: 1/3 correct
  expect_equal(as.numeric(result$baseline_lookup[["1"]]), 2/3, tolerance = 1e-6)
  expect_equal(as.numeric(result$baseline_lookup[["2"]]), 1/3, tolerance = 1e-6)
  expect_equal(result$global_mean, 0.5, tolerance = 1e-6)
})

test_that("compute_item_baseline works with shrinkage", {
  y_train <- c(1, 1, 0, 1, 0, 0)
  item_id_train <- c(1, 1, 1, 2, 2, 2)
  alpha <- 2.0
  beta <- 2.0

  result <- compute_item_baseline(
    y_train, item_id_train, use_shrinkage = TRUE, alpha = alpha, beta = beta
  )

  # Item 1: (2 + 2) / (3 + 2 + 2) = 4/7
  # Item 2: (1 + 2) / (3 + 2 + 2) = 3/7
  expect_equal(as.numeric(result$baseline_lookup[["1"]]), 4/7, tolerance = 1e-6)
  expect_equal(as.numeric(result$baseline_lookup[["2"]]), 3/7, tolerance = 1e-6)
})

test_that("shrinkage pulls extreme values toward prior", {
  y_train <- c(1, 1)
  item_id_train <- c(1, 1)

  result_no_shrink <- compute_item_baseline(y_train, item_id_train, use_shrinkage = FALSE)
  result_shrink <- compute_item_baseline(
    y_train, item_id_train, use_shrinkage = TRUE, alpha = 2.0, beta = 2.0
  )

  # Without shrinkage: 1.0
  # With shrinkage: (2 + 2) / (2 + 2 + 2) = 4/6 < 1.0
  expect_equal(as.numeric(result_no_shrink$baseline_lookup[["1"]]), 1.0)
  expect_lt(as.numeric(result_shrink$baseline_lookup[["1"]]), 1.0)
})

test_that("compute_item_baseline handles multiple items", {
  y_train <- c(1, 0, 1, 1, 0, 0, 1, 1, 1)
  item_id_train <- c(1, 1, 1, 2, 2, 2, 3, 3, 3)

  result <- compute_item_baseline(y_train, item_id_train, use_shrinkage = FALSE)

  expect_length(result$baseline_lookup, 3)
  expect_equal(as.numeric(result$baseline_lookup[["1"]]), 2/3, tolerance = 1e-6)
  expect_equal(as.numeric(result$baseline_lookup[["2"]]), 1/3, tolerance = 1e-6)
  expect_equal(as.numeric(result$baseline_lookup[["3"]]), 1.0, tolerance = 1e-6)
  expect_equal(result$global_mean, 6/9, tolerance = 1e-6)
})

# ==============================================================================
# Tests for map_item_baseline
# ==============================================================================

test_that("map_item_baseline performs basic mapping", {
  item_id_eval <- c(1, 2, 2, 3, 3)
  baseline_probs <- c("1" = 0.9, "2" = 0.4, "3" = 0.5)
  global_mean <- 0.6

  result <- map_item_baseline(item_id_eval, baseline_probs, global_mean)

  expected <- c(0.9, 0.4, 0.4, 0.5, 0.5)
  expect_equal(result, expected, tolerance = 1e-5)
})

test_that("map_item_baseline falls back to global mean for unknown items", {
  item_id_eval <- c(1, 2, 99)  # 99 not in lookup
  baseline_probs <- c("1" = 0.9, "2" = 0.4)
  global_mean <- 0.6

  result <- map_item_baseline(item_id_eval, baseline_probs, global_mean)

  expect_equal(result[3], global_mean, tolerance = 1e-5)
})

test_that("map_item_baseline handles empty input", {
  item_id_eval <- integer(0)
  baseline_probs <- c("1" = 0.9)
  global_mean <- 0.6

  result <- map_item_baseline(item_id_eval, baseline_probs, global_mean)

  expect_length(result, 0)
})

test_that("map_item_baseline handles empty baseline_probs", {
  item_id_eval <- c(1, 2, 3)
  baseline_probs <- numeric(0)
  global_mean <- 0.6

  result <- map_item_baseline(item_id_eval, baseline_probs, global_mean)

  # All should be global_mean
  expect_equal(result, c(0.6, 0.6, 0.6), tolerance = 1e-5)
})

test_that("map_item_baseline clips values", {
  eps <- 0.01
  item_id_eval <- c(1, 2)
  baseline_probs <- c("1" = 0.0, "2" = 1.0)
  global_mean <- 0.5

  result <- map_item_baseline(item_id_eval, baseline_probs, global_mean, eps = eps)

  expect_gte(result[1], eps - 1e-6)
  expect_lte(result[2], 1 - eps + 1e-6)
})

# ==============================================================================
# Tests for check_calibration
# ==============================================================================

test_that("check_calibration computes ECE for calibrated predictions", {
  set.seed(42)
  n <- 1000
  y_pred <- runif(n)
  y_true <- as.numeric(runif(n) < y_pred)

  result <- check_calibration(y_pred, y_true, n_bins = 10, warn_threshold = 0.5)

  # ECE should be relatively low for calibrated predictions
  expect_lt(result$ece, 0.1)
  expect_true("bin_accuracies" %in% names(result))
  expect_true("bin_confidences" %in% names(result))
  expect_true("bin_counts" %in% names(result))
})

test_that("check_calibration detects overconfident predictions", {
  # All predictions are 0.9, but only 50% are correct
  y_pred <- rep(0.9, 100)
  y_true <- c(rep(1, 50), rep(0, 50))

  result <- check_calibration(y_pred, y_true, n_bins = 10, warn_threshold = 0.5)

  # ECE should be high (0.9 - 0.5 = 0.4)
  expect_gt(result$ece, 0.3)
})

test_that("check_calibration detects underconfident predictions", {
  # All predictions are 0.5, but 90% are correct
  y_pred <- rep(0.5, 100)
  y_true <- c(rep(1, 90), rep(0, 10))

  result <- check_calibration(y_pred, y_true, n_bins = 10, warn_threshold = 0.5)

  # ECE should be high (0.9 - 0.5 = 0.4)
  expect_gt(result$ece, 0.3)
})

test_that("check_calibration warns when threshold exceeded", {
  y_pred <- rep(0.9, 100)
  y_true <- c(rep(1, 50), rep(0, 50))

  expect_warning(
    check_calibration(y_pred, y_true, n_bins = 10, warn_threshold = 0.1),
    "ECE"
  )
})

test_that("check_calibration returns correct structure", {
  y_pred <- c(0.1, 0.5, 0.9)
  y_true <- c(0, 1, 1)

  result <- check_calibration(y_pred, y_true, n_bins = 10, warn_threshold = 0.5)

  expect_type(result$ece, "double")
  expect_type(result$bin_accuracies, "double")
  expect_type(result$bin_confidences, "double")
  expect_type(result$bin_counts, "integer")
})

# ==============================================================================
# Tests for fractional_information_gain_validation - Basic Behavior
# ==============================================================================

test_that("perfect model has high FIG-V", {
  set.seed(42)
  n_students <- 50
  n_items <- 10
  n_obs <- 500

  student_ids <- sample(1:n_students, n_obs, replace = TRUE)
  item_ids <- sample(1:n_items, n_obs, replace = TRUE)
  y_train <- sample(c(0, 1), n_obs, replace = TRUE, prob = c(0.4, 0.6))
  item_id_train <- sample(1:n_items, n_obs, replace = TRUE)
  y_eval <- sample(c(0, 1), n_obs, replace = TRUE, prob = c(0.4, 0.6))

  # Perfect predictions
  y_pred <- pmax(pmin(as.numeric(y_eval), 0.999), 0.001)

  result <- fractional_information_gain_validation(
    y_pred_eval = y_pred,
    y_eval = y_eval,
    item_id_eval = item_ids,
    student_id_eval = student_ids,
    y_train = y_train,
    item_id_train = item_id_train
  )

  expect_gt(result$fig_v, 0.8)
  expect_gt(result$fig_v_pooled, 0.8)
  expect_lte(result$fig_v, 1 + 1e-6)
  expect_lte(result$fig_v_pooled, 1 + 1e-6)
})

test_that("random model has low FIG-V", {
  set.seed(42)
  n_students <- 50
  n_items <- 10
  n_obs <- 500

  student_ids <- sample(1:n_students, n_obs, replace = TRUE)
  item_ids <- sample(1:n_items, n_obs, replace = TRUE)
  y_train <- sample(c(0, 1), n_obs, replace = TRUE, prob = c(0.4, 0.6))
  item_id_train <- sample(1:n_items, n_obs, replace = TRUE)
  y_eval <- sample(c(0, 1), n_obs, replace = TRUE, prob = c(0.4, 0.6))

  # Random predictions (all 0.5)
  y_pred <- rep(0.5, n_obs)

  result <- fractional_information_gain_validation(
    y_pred_eval = y_pred,
    y_eval = y_eval,
    item_id_eval = item_ids,
    student_id_eval = student_ids,
    y_train = y_train,
    item_id_train = item_id_train
  )

  expect_lt(result$fig_v, 0.3)
  expect_lt(result$fig_v_pooled, 0.3)
})

test_that("baseline model has FIG-V near 0", {
  set.seed(42)
  n_students <- 50
  n_items <- 5
  n_train <- 1000
  n_eval <- 500

  # Create items with distinct difficulties
  item_difficulties <- c(0.9, 0.7, 0.5, 0.3, 0.1)
  names(item_difficulties) <- as.character(1:5)

  # Training data
  item_id_train <- sample(1:5, n_train, replace = TRUE)
  y_train <- sapply(item_id_train, function(item) rbinom(1, 1, item_difficulties[item]))

  # Eval data
  student_ids <- sample(1:n_students, n_eval, replace = TRUE)
  item_id_eval <- sample(1:5, n_eval, replace = TRUE)
  y_eval <- sapply(item_id_eval, function(item) rbinom(1, 1, item_difficulties[item]))

  # Predict using item base rates
  item_means <- tapply(y_train, item_id_train, mean)
  y_pred <- item_means[as.character(item_id_eval)]

  result <- fractional_information_gain_validation(
    y_pred_eval = y_pred,
    y_eval = y_eval,
    item_id_eval = item_id_eval,
    student_id_eval = student_ids,
    y_train = y_train,
    item_id_train = item_id_train,
    use_shrinkage = FALSE
  )

  expect_lt(abs(result$fig_v), 0.15)
  expect_lt(abs(result$fig_v_pooled), 0.15)
})

# ==============================================================================
# Tests for fractional_information_gain_validation - Edge Cases
# ==============================================================================

test_that("FIG-V works with single student", {
  y_pred <- c(0.8, 0.6, 0.7, 0.9)
  y_eval <- c(1, 0, 1, 1)
  item_id_eval <- c(1, 2, 1, 2)
  student_id_eval <- c(1, 1, 1, 1)  # Single student
  y_train <- c(1, 0, 1, 0)
  item_id_train <- c(1, 1, 2, 2)

  result <- fractional_information_gain_validation(
    y_pred_eval = y_pred,
    y_eval = y_eval,
    item_id_eval = item_id_eval,
    student_id_eval = student_id_eval,
    y_train = y_train,
    item_id_train = item_id_train
  )

  expect_true("fig_v" %in% names(result))
  expect_true("fig_v_pooled" %in% names(result))
  expect_true(is.finite(result$fig_v))
})

test_that("FIG-V works with single item", {
  y_pred <- c(0.8, 0.6, 0.7, 0.9)
  y_eval <- c(1, 0, 1, 1)
  item_id_eval <- c(1, 1, 1, 1)  # Single item
  student_id_eval <- c(1, 2, 3, 4)
  y_train <- c(1, 0, 1, 0)
  item_id_train <- c(1, 1, 1, 1)

  result <- fractional_information_gain_validation(
    y_pred_eval = y_pred,
    y_eval = y_eval,
    item_id_eval = item_id_eval,
    student_id_eval = student_id_eval,
    y_train = y_train,
    item_id_train = item_id_train
  )

  expect_true("fig_v" %in% names(result))
  expect_true(is.finite(result$fig_v))
})

test_that("FIG-V works with all correct responses", {
  y_pred <- c(0.9, 0.8, 0.85, 0.95)
  y_eval <- c(1, 1, 1, 1)
  item_id_eval <- c(1, 2, 1, 2)
  student_id_eval <- c(1, 1, 2, 2)
  y_train <- c(1, 0, 1, 0)
  item_id_train <- c(1, 1, 2, 2)

  result <- fractional_information_gain_validation(
    y_pred_eval = y_pred,
    y_eval = y_eval,
    item_id_eval = item_id_eval,
    student_id_eval = student_id_eval,
    y_train = y_train,
    item_id_train = item_id_train
  )

  expect_true("fig_v" %in% names(result))
  expect_true(is.finite(result$fig_v))
})

test_that("FIG-V works with all incorrect responses", {
  y_pred <- c(0.1, 0.2, 0.15, 0.05)
  y_eval <- c(0, 0, 0, 0)
  item_id_eval <- c(1, 2, 1, 2)
  student_id_eval <- c(1, 1, 2, 2)
  y_train <- c(1, 0, 1, 0)
  item_id_train <- c(1, 1, 2, 2)

  result <- fractional_information_gain_validation(
    y_pred_eval = y_pred,
    y_eval = y_eval,
    item_id_eval = item_id_eval,
    student_id_eval = student_id_eval,
    y_train = y_train,
    item_id_train = item_id_train
  )

  expect_true("fig_v" %in% names(result))
  expect_true(is.finite(result$fig_v))
})

test_that("FIG-V handles unseen items in eval", {
  y_pred <- c(0.8, 0.6, 0.7)
  y_eval <- c(1, 0, 1)
  item_id_eval <- c(1, 2, 99)  # Item 99 not in train
  student_id_eval <- c(1, 1, 2)
  y_train <- c(1, 0)
  item_id_train <- c(1, 2)

  result <- fractional_information_gain_validation(
    y_pred_eval = y_pred,
    y_eval = y_eval,
    item_id_eval = item_id_eval,
    student_id_eval = student_id_eval,
    y_train = y_train,
    item_id_train = item_id_train
  )

  expect_true("fig_v" %in% names(result))
  expect_true(is.finite(result$fig_v))
})

# ==============================================================================
# Tests for fractional_information_gain_validation - Calibration
# ==============================================================================

test_that("calibration output is returned when requested", {
  set.seed(42)
  n <- 100

  y_pred <- runif(n, 0.1, 0.9)
  y_eval <- sample(c(0, 1), n, replace = TRUE)
  item_id_eval <- sample(1:5, n, replace = TRUE)
  student_id_eval <- sample(1:10, n, replace = TRUE)
  y_train <- sample(c(0, 1), n, replace = TRUE)
  item_id_train <- sample(1:5, n, replace = TRUE)

  result <- fractional_information_gain_validation(
    y_pred_eval = y_pred,
    y_eval = y_eval,
    item_id_eval = item_id_eval,
    student_id_eval = student_id_eval,
    y_train = y_train,
    item_id_train = item_id_train,
    calibration = TRUE
  )

  expect_true("calibration" %in% names(result))
  expect_true("ece" %in% names(result$calibration))
  expect_type(result$calibration$ece, "double")
})

test_that("calibration key is present but NULL when not requested", {
  y_pred <- c(0.8, 0.6, 0.7)
  y_eval <- c(1, 0, 1)
  item_id_eval <- c(1, 2, 1)
  student_id_eval <- c(1, 1, 2)
  y_train <- c(1, 0)
  item_id_train <- c(1, 2)

  result <- fractional_information_gain_validation(
    y_pred_eval = y_pred,
    y_eval = y_eval,
    item_id_eval = item_id_eval,
    student_id_eval = student_id_eval,
    y_train = y_train,
    item_id_train = item_id_train,
    calibration = FALSE
  )

  # Mirrors the Python return shape: the key is always present, valued NULL
  # when calibration was not requested.
  expect_true("calibration" %in% names(result))
  expect_null(result$calibration)
})

# ==============================================================================
# Tests for fractional_information_gain_validation - Consistency
# ==============================================================================

test_that("output is deterministic", {
  y_pred <- c(0.8, 0.6, 0.7, 0.9)
  y_eval <- c(1, 0, 1, 1)
  item_id_eval <- c(1, 2, 1, 2)
  student_id_eval <- c(1, 1, 2, 2)
  y_train <- c(1, 0, 1, 0)
  item_id_train <- c(1, 1, 2, 2)

  result1 <- fractional_information_gain_validation(
    y_pred_eval = y_pred,
    y_eval = y_eval,
    item_id_eval = item_id_eval,
    student_id_eval = student_id_eval,
    y_train = y_train,
    item_id_train = item_id_train
  )

  result2 <- fractional_information_gain_validation(
    y_pred_eval = y_pred,
    y_eval = y_eval,
    item_id_eval = item_id_eval,
    student_id_eval = student_id_eval,
    y_train = y_train,
    item_id_train = item_id_train
  )

  expect_equal(result1$fig_v, result2$fig_v)
  expect_equal(result1$fig_v_pooled, result2$fig_v_pooled)
})

test_that("FIG-V denominator uses Y items with train baselines", {
  item_id_train <- c(rep(1, 8), rep(2, 2))
  y_train <- c(1, 1, 1, 1, 1, 1, 0, 0, 1, 0)

  item_id_eval <- c(rep(1, 2), rep(2, 8))
  y_eval <- c(1, 0, 1, 0, 1, 0, 1, 0, 1, 0)
  student_id_eval <- c(1, 1, 2, 2, 3, 3, 4, 4, 5, 5)

  y_pred <- rep(0.5, length(y_eval))

  result <- fractional_information_gain_validation(
    y_pred_eval = y_pred,
    y_eval = y_eval,
    item_id_eval = item_id_eval,
    student_id_eval = student_id_eval,
    y_train = y_train,
    item_id_train = item_id_train,
    use_shrinkage = FALSE
  )

  # Uses the package's binary_entropy (exposed via pkgload::load_all).
  entropy_y <- (2 * binary_entropy(0.75) + 8 * binary_entropy(0.5)) / 10
  expected <- 1 - log(2) / entropy_y

  expect_equal(result$fig_v_pooled, expected, tolerance = 1e-8)
})

test_that("shrinkage vs no shrinkage produces valid results", {
  set.seed(42)
  n <- 100

  y_pred <- runif(n, 0.2, 0.8)
  y_eval <- sample(c(0, 1), n, replace = TRUE)
  item_id_eval <- sample(1:5, n, replace = TRUE)
  student_id_eval <- sample(1:10, n, replace = TRUE)
  y_train <- sample(c(0, 1), n, replace = TRUE)
  item_id_train <- sample(1:5, n, replace = TRUE)

  result_shrink <- fractional_information_gain_validation(
    y_pred_eval = y_pred,
    y_eval = y_eval,
    item_id_eval = item_id_eval,
    student_id_eval = student_id_eval,
    y_train = y_train,
    item_id_train = item_id_train,
    use_shrinkage = TRUE
  )

  result_no_shrink <- fractional_information_gain_validation(
    y_pred_eval = y_pred,
    y_eval = y_eval,
    item_id_eval = item_id_eval,
    student_id_eval = student_id_eval,
    y_train = y_train,
    item_id_train = item_id_train,
    use_shrinkage = FALSE
  )

  expect_true(is.finite(result_shrink$fig_v))
  expect_true(is.finite(result_no_shrink$fig_v))
})

# ==============================================================================
# Tests for fractional_information_gain_validation - Bounds
# ==============================================================================

test_that("FIG-V is always at most one", {
  set.seed(42)

  for (i in 1:10) {
    n <- sample(50:200, 1)
    n_items <- sample(3:10, 1)
    n_students <- sample(5:20, 1)

    y_pred <- runif(n, 0.1, 0.9)
    y_eval <- sample(c(0, 1), n, replace = TRUE)
    item_id_eval <- sample(1:n_items, n, replace = TRUE)
    student_id_eval <- sample(1:n_students, n, replace = TRUE)
    y_train <- sample(c(0, 1), n, replace = TRUE)
    item_id_train <- sample(1:n_items, n, replace = TRUE)

    result <- fractional_information_gain_validation(
      y_pred_eval = y_pred,
      y_eval = y_eval,
      item_id_eval = item_id_eval,
      student_id_eval = student_id_eval,
      y_train = y_train,
      item_id_train = item_id_train
    )

    expect_lte(result$fig_v, 1 + 1e-6, label = paste("fig_v =", result$fig_v))
    expect_lte(result$fig_v_pooled, 1 + 1e-6, label = paste("fig_v_pooled =", result$fig_v_pooled))
  }
})

test_that("FIG-V can be negative when model is worse than baseline", {
  # Predict opposite of truth
  y_pred <- c(0.1, 0.1, 0.9, 0.9)
  y_eval <- c(1, 1, 0, 0)
  item_id_eval <- c(1, 2, 1, 2)
  student_id_eval <- c(1, 1, 2, 2)
  y_train <- c(1, 1, 0, 0)
  item_id_train <- c(1, 1, 2, 2)

  result <- fractional_information_gain_validation(
    y_pred_eval = y_pred,
    y_eval = y_eval,
    item_id_eval = item_id_eval,
    student_id_eval = student_id_eval,
    y_train = y_train,
    item_id_train = item_id_train
  )

  expect_lt(result$fig_v, 0, label = paste("Expected negative fig_v, got", result$fig_v))
  expect_lte(result$fig_v, 1 + 1e-6)
})

test_that("perfect model approaches one", {
  set.seed(42)
  n <- 500

  y_eval <- sample(c(0, 1), n, replace = TRUE)
  y_pred <- ifelse(y_eval == 1, 0.99, 0.01)
  item_id_eval <- sample(1:10, n, replace = TRUE)
  student_id_eval <- sample(1:20, n, replace = TRUE)
  y_train <- sample(c(0, 1), n, replace = TRUE)
  item_id_train <- sample(1:10, n, replace = TRUE)

  result <- fractional_information_gain_validation(
    y_pred_eval = y_pred,
    y_eval = y_eval,
    item_id_eval = item_id_eval,
    student_id_eval = student_id_eval,
    y_train = y_train,
    item_id_train = item_id_train
  )

  expect_gt(result$fig_v, 0.9, label = paste("Expected high fig_v, got", result$fig_v))
  expect_lte(result$fig_v, 1 + 1e-6)
  expect_gt(result$fig_v_pooled, 0.9)
  expect_lte(result$fig_v_pooled, 1 + 1e-6)
})

# ==============================================================================
# Tests for Model with Student Effects
# ==============================================================================

test_that("model capturing student ability outperforms baseline", {
  set.seed(123)
  n_students <- 50
  n_items <- 10
  responses_per_student <- 30
  n_eval <- n_students * responses_per_student

  # Generate student abilities with wider variance
  student_abilities <- rnorm(n_students + 1, 0, 1.5)
  item_difficulties <- runif(n_items + 1, -1, 1)

  # Training data
  n_train <- 2000
  item_id_train <- sample(1:n_items, n_train, replace = TRUE)
  y_train <- sapply(item_id_train, function(item) {
    rbinom(1, 1, 1 / (1 + exp(-item_difficulties[item])))
  })

  # Eval data with student effects
  student_ids <- rep(1:n_students, each = responses_per_student)
  item_id_eval <- sample(1:n_items, n_eval, replace = TRUE)

  true_probs <- mapply(function(s, i) {
    1 / (1 + exp(-(student_abilities[s] + item_difficulties[i])))
  }, student_ids, item_id_eval)

  y_eval <- as.numeric(runif(n_eval) < true_probs)

  # Good model: knows student abilities
  y_pred_good <- pmax(pmin(true_probs, 0.99), 0.01)

  # Bad model: constant prediction
  y_pred_bad <- rep(0.5, n_eval)

  result_good <- fractional_information_gain_validation(
    y_pred_eval = y_pred_good,
    y_eval = y_eval,
    item_id_eval = item_id_eval,
    student_id_eval = student_ids,
    y_train = y_train,
    item_id_train = item_id_train
  )

  result_bad <- fractional_information_gain_validation(
    y_pred_eval = y_pred_bad,
    y_eval = y_eval,
    item_id_eval = item_id_eval,
    student_id_eval = student_ids,
    y_train = y_train,
    item_id_train = item_id_train
  )

  expect_gt(result_good$fig_v, result_bad$fig_v)
  expect_lte(result_good$fig_v, 1 + 1e-6)
  expect_lte(result_good$fig_v_pooled, 1 + 1e-6)
  expect_lte(result_bad$fig_v, 1 + 1e-6)
  expect_lte(result_bad$fig_v_pooled, 1 + 1e-6)
})

# ==============================================================================
# Tests for binary_entropy
# ==============================================================================

test_that("binary entropy is maximum at p=0.5", {
  p <- 0.5
  result <- binary_entropy(p)
  expected <- log(2)  # Maximum entropy for binary
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("binary entropy is near zero at extremes", {
  eps <- 1e-12
  p <- c(eps, 1 - eps)
  result <- binary_entropy(p)
  expect_lt(result[1], 0.001)
  expect_lt(result[2], 0.001)
})

test_that("binary entropy is symmetric", {
  p <- c(0.2, 0.3, 0.4)
  result_p <- binary_entropy(p)
  result_1_minus_p <- binary_entropy(1 - p)
  expect_equal(result_p, result_1_minus_p, tolerance = 1e-10)
})

# ==============================================================================
# Tests for fractional_information_gain_confidence (FIG-C)
# ==============================================================================

test_that("FIG-C basic computation works", {
  y_pred <- c(0.9, 0.1, 0.8, 0.2)
  item_id_eval <- c(1, 1, 2, 2)
  student_id_eval <- c(1, 1, 2, 2)
  y_train <- c(1, 1, 0, 0)
  item_id_train <- c(1, 1, 2, 2)

  result <- fractional_information_gain_confidence(
    y_pred_eval = y_pred,
    item_id_eval = item_id_eval,
    student_id_eval = student_id_eval,
    y_train = y_train,
    item_id_train = item_id_train
  )

  expect_true("fig_c" %in% names(result))
  expect_true("fig_c_pooled" %in% names(result))
  expect_true("fig_c_by_student" %in% names(result))
  expect_true(is.finite(result$fig_c))
  expect_true(is.finite(result$fig_c_pooled))
})

test_that("confident model has high FIG-C", {
  # Model is very confident (predictions near 0 or 1)
  y_pred <- c(0.99, 0.01, 0.99, 0.01)
  item_id_eval <- c(1, 1, 2, 2)
  student_id_eval <- c(1, 1, 2, 2)
  y_train <- c(1, 0, 1, 0, 1, 0)
  item_id_train <- c(1, 1, 1, 2, 2, 2)

  result <- fractional_information_gain_confidence(
    y_pred_eval = y_pred,
    item_id_eval = item_id_eval,
    student_id_eval = student_id_eval,
    y_train = y_train,
    item_id_train = item_id_train
  )

  # Model entropy is low (confident), so FIG-C should be high
  expect_gt(result$fig_c, 0.5)
  expect_gt(result$fig_c_pooled, 0.5)
})

test_that("uncertain model has low FIG-C", {
  # Model is uncertain (predictions near 0.5)
  y_pred <- c(0.5, 0.5, 0.5, 0.5)
  item_id_eval <- c(1, 1, 2, 2)
  student_id_eval <- c(1, 1, 2, 2)
  # Training has items with 50% base rate too
  y_train <- c(1, 0, 1, 0)
  item_id_train <- c(1, 1, 2, 2)

  result <- fractional_information_gain_confidence(
    y_pred_eval = y_pred,
    item_id_eval = item_id_eval,
    student_id_eval = student_id_eval,
    y_train = y_train,
    item_id_train = item_id_train,
    use_shrinkage = FALSE  # No shrinkage so baseline is 0.5
  )

  # Model entropy equals baseline entropy, so FIG-C should be ~0
  expect_lt(abs(result$fig_c), 0.1)
  expect_lt(abs(result$fig_c_pooled), 0.1)
})

test_that("FIG-C is bounded above by 1", {
  set.seed(42)
  for (i in 1:10) {
    n <- sample(50:100, 1)
    y_pred <- runif(n, 0.01, 0.99)
    item_id_eval <- sample(1:5, n, replace = TRUE)
    student_id_eval <- sample(1:10, n, replace = TRUE)
    y_train <- sample(c(0, 1), n * 2, replace = TRUE)
    item_id_train <- sample(1:5, n * 2, replace = TRUE)

    result <- fractional_information_gain_confidence(
      y_pred_eval = y_pred,
      item_id_eval = item_id_eval,
      student_id_eval = student_id_eval,
      y_train = y_train,
      item_id_train = item_id_train
    )

    expect_lte(result$fig_c, 1 + 1e-6)
    expect_lte(result$fig_c_pooled, 1 + 1e-6)
  }
})

test_that("FIG-C works with single student", {
  y_pred <- c(0.9, 0.8, 0.7)
  item_id_eval <- c(1, 2, 3)
  student_id_eval <- c(1, 1, 1)
  y_train <- c(1, 0, 1, 0, 1, 0)
  item_id_train <- c(1, 1, 2, 2, 3, 3)

  result <- fractional_information_gain_confidence(
    y_pred_eval = y_pred,
    item_id_eval = item_id_eval,
    student_id_eval = student_id_eval,
    y_train = y_train,
    item_id_train = item_id_train
  )

  expect_true("fig_c" %in% names(result))
  expect_true(is.finite(result$fig_c))
})

test_that("FIG-C deterministic output", {
  y_pred <- c(0.8, 0.6, 0.7, 0.9)
  item_id_eval <- c(1, 2, 1, 2)
  student_id_eval <- c(1, 1, 2, 2)
  y_train <- c(1, 0, 1, 0)
  item_id_train <- c(1, 1, 2, 2)

  result1 <- fractional_information_gain_confidence(
    y_pred_eval = y_pred,
    item_id_eval = item_id_eval,
    student_id_eval = student_id_eval,
    y_train = y_train,
    item_id_train = item_id_train
  )

  result2 <- fractional_information_gain_confidence(
    y_pred_eval = y_pred,
    item_id_eval = item_id_eval,
    student_id_eval = student_id_eval,
    y_train = y_train,
    item_id_train = item_id_train
  )

  expect_equal(result1$fig_c, result2$fig_c)
  expect_equal(result1$fig_c_pooled, result2$fig_c_pooled)
})

test_that("FIG-C student_ids and fig_c_by_student are in consistent order", {
  # Students appear in non-sorted order (insertion order: 3, 1, 2) to exercise
  # the case where unique() and tapply() would produce different orderings.
  # student_ids[i] must correspond to fig_c_by_student[i] at every position.
  y_pred          <- c(0.9, 0.2, 0.8, 0.1, 0.7, 0.3)
  item_id_eval    <- c(1,   1,   2,   2,   1,   2  )
  student_id_eval <- c(3,   1,   2,   3,   1,   2  )
  y_train         <- c(1, 0, 1, 0)
  item_id_train   <- c(1, 1, 2, 2)

  result <- fractional_information_gain_confidence(
    y_pred_eval     = y_pred,
    item_id_eval    = item_id_eval,
    student_id_eval = student_id_eval,
    y_train         = y_train,
    item_id_train   = item_id_train,
    use_shrinkage   = FALSE
  )

  # student_ids must be consistent with the names of fig_c_by_student.
  expect_equal(
    as.character(result$student_ids),
    names(result$fig_c_by_student)
  )

  # Verify the per-student values independently by hand.
  # With use_shrinkage = FALSE: item 1 baseline = 1/2, item 2 baseline = 1/2,
  # so entropy_baseline = log(2) for every observation.
  log2_val <- log(2)
  expected_s1 <- 1 - (binary_entropy(0.2) + binary_entropy(0.7)) / (2 * log2_val)
  expected_s2 <- 1 - (binary_entropy(0.8) + binary_entropy(0.3)) / (2 * log2_val)
  expected_s3 <- 1 - (binary_entropy(0.9) + binary_entropy(0.1)) / (2 * log2_val)

  expect_equal(result$fig_c_by_student[["1"]], expected_s1, tolerance = 1e-10)
  expect_equal(result$fig_c_by_student[["2"]], expected_s2, tolerance = 1e-10)
  expect_equal(result$fig_c_by_student[["3"]], expected_s3, tolerance = 1e-10)
})

# ==============================================================================
# Tests for FIG-V per-student outputs (parity with FIG-C and the Python version)
# ==============================================================================

test_that("FIG-V returns per-student values with consistent student_ids", {
  # Students appear out of order (3, 1, 2) to exercise the sorting in tapply().
  y_pred          <- c(0.9, 0.2, 0.8, 0.1, 0.7, 0.3)
  y_eval          <- c(1,   0,   1,   0,   1,   0  )
  item_id_eval    <- c(1,   1,   2,   2,   1,   2  )
  student_id_eval <- c(3,   1,   2,   3,   1,   2  )
  y_train         <- c(1, 0, 1, 0)
  item_id_train   <- c(1, 1, 2, 2)

  result <- fractional_information_gain_validation(
    y_pred_eval     = y_pred,
    y_eval          = y_eval,
    item_id_eval    = item_id_eval,
    student_id_eval = student_id_eval,
    y_train         = y_train,
    item_id_train   = item_id_train,
    use_shrinkage   = FALSE
  )

  # The per-student outputs are present and internally consistent.
  expect_true("fig_v_by_student" %in% names(result))
  expect_true("student_ids" %in% names(result))
  expect_equal(
    as.character(result$student_ids),
    names(result$fig_v_by_student)
  )
  # Student-weighted fig_v is the mean of the per-student values.
  expect_equal(result$fig_v, mean(result$fig_v_by_student))

  # With use_shrinkage = FALSE both items have baseline 0.5, so every
  # observation's baseline entropy is log(2). Verify per-student values by hand.
  bce <- function(y, p) -(y * log(p) + (1 - y) * log(1 - p))
  log2_val <- log(2)
  expected_s1 <- 1 - (bce(0, 0.2) + bce(1, 0.7)) / (2 * log2_val)
  expected_s2 <- 1 - (bce(1, 0.8) + bce(0, 0.3)) / (2 * log2_val)
  expected_s3 <- 1 - (bce(1, 0.9) + bce(0, 0.1)) / (2 * log2_val)

  expect_equal(result$fig_v_by_student[["1"]], expected_s1, tolerance = 1e-10)
  expect_equal(result$fig_v_by_student[["2"]], expected_s2, tolerance = 1e-10)
  expect_equal(result$fig_v_by_student[["3"]], expected_s3, tolerance = 1e-10)
})

# ==============================================================================
# Tests for input validation: non-integer IDs and empty inputs
# ==============================================================================

test_that("non-integer IDs raise a clear error", {
  base_args <- list(
    y_pred_eval = c(0.8, 0.6),
    y_eval = c(1, 0),
    item_id_eval = c(1, 2),
    student_id_eval = c(1, 1),
    y_train = c(1, 0),
    item_id_train = c(1, 2)
  )

  # Fractional item IDs would be silently truncated by as.integer(); error instead.
  args <- base_args
  args$item_id_eval <- c(1.5, 2.5)
  expect_error(
    do.call(fractional_information_gain_validation, args), "integer"
  )

  # Non-numeric student IDs would become NA; error instead.
  args <- base_args
  args$student_id_eval <- c("a", "b")
  expect_error(
    do.call(fractional_information_gain_validation, args), "integer"
  )
})

test_that("empty inputs raise an error (FIG-V and FIG-C)", {
  expect_error(
    fractional_information_gain_validation(
      y_pred_eval = numeric(0),
      y_eval = numeric(0),
      item_id_eval = integer(0),
      student_id_eval = integer(0),
      y_train = c(1, 0),
      item_id_train = c(1, 2)
    ),
    "empty"
  )

  expect_error(
    fractional_information_gain_confidence(
      y_pred_eval = numeric(0),
      item_id_eval = integer(0),
      student_id_eval = integer(0),
      y_train = c(1, 0),
      item_id_train = c(1, 2)
    ),
    "empty"
  )
})
