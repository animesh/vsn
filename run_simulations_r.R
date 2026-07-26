#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1L) args[[1L]] else "vsn_simulation_results"
manifest_path <- file.path(root, "manifest.tsv")
r_output_dir <- file.path(root, "r")
dir.create(r_output_dir, recursive = TRUE, showWarnings = FALSE)
options(digits = 17, scipen = 999)

if (!requireNamespace("vsn", quietly = TRUE)) {
  stop("Bioconductor package 'vsn' is required.")
}

manifest <- read.delim(
  manifest_path,
  stringsAsFactors = FALSE,
  check.names = FALSE,
  quote = "",
  comment.char = ""
)

run_rows <- vector("list", nrow(manifest))

for (i in seq_len(nrow(manifest))) {
  scenario <- manifest$scenario[[i]]
  input_path <- manifest$input_file[[i]]
  message("R scenario: ", scenario)

  dat <- read.delim(
    input_path,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    quote = "",
    comment.char = "",
    na.strings = c("NA", "NaN", "")
  )
  intensity_columns <- grep("^LFQ intensity ", colnames(dat), value = TRUE)
  x <- as.matrix(dat[, intensity_columns, drop = FALSE])
  storage.mode(x) <- "double"
  x[!is.finite(x)] <- NA_real_
  x[x == 0] <- NA_real_

  error_message <- ""
  elapsed <- NA_real_
  finite_output <- 0L
  fit_fail <- NA_integer_

  start <- proc.time()[["elapsed"]]
  normalized <- tryCatch(
    {
      fit <- vsn::vsn2(
        x,
        lts.quantile = manifest$lts_quantile[[i]],
        subsample = 0L,
        verbose = FALSE,
        returnData = TRUE,
        calib = manifest$calib[[i]],
        minDataPointsPerStratum = 0L
      )
      fit_fail <- as.integer(fit@lbfgsb)
      stats::predict(fit, newdata = x, useDataInFit = TRUE)
    },
    error = function(e) {
      error_message <<- paste(class(e)[[1L]], conditionMessage(e), sep = ": ")
      NULL
    }
  )
  elapsed <- proc.time()[["elapsed"]] - start

  if (!is.null(normalized)) {
    storage.mode(normalized) <- "double"
    finite_output <- sum(is.finite(normalized))
    out <- data.frame(
      feature_id = dat$feature_id,
      normalized,
      check.names = FALSE
    )
    colnames(out)[-1L] <- paste0("vsn_", intensity_columns)
    write.table(
      out,
      file = file.path(r_output_dir, paste0(scenario, ".r.tsv")),
      sep = "\t",
      row.names = FALSE,
      col.names = TRUE,
      quote = FALSE,
      na = "NA"
    )
  }

  finite_values <- x[is.finite(x)]
  observed_per_row <- rowSums(is.finite(x))
  run_rows[[i]] <- data.frame(
    scenario = scenario,
    rows = nrow(x),
    samples = ncol(x),
    finite_input_values = sum(is.finite(x)),
    actual_missing_fraction = mean(!is.finite(x)),
    all_na_rows = sum(observed_per_row == 0L),
    rows_with_one_value = sum(observed_per_row == 1L),
    minimum_finite_value = if (length(finite_values)) min(finite_values) else NA_real_,
    maximum_finite_value = if (length(finite_values)) max(finite_values) else NA_real_,
    optimizer_return_code = fit_fail,
    finite_output_values = finite_output,
    elapsed_seconds = elapsed,
    error = error_message,
    stringsAsFactors = FALSE
  )
}

write.table(
  do.call(rbind, run_rows),
  file = file.path(root, "r_run_summary.tsv"),
  sep = "\t", row.names = FALSE, quote = FALSE, na = "NA"
)

capture.output(sessionInfo(), file = file.path(root, "r_session_info.txt"))
