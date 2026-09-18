# Reproduce the 2018 Coral.Forest random forest (original/How2TrainYourCoralBleachingAI.Rmd).
#
# The cleaning steps and model call are copied from the 2018 notebook. Three
# things differ, and none of them change the method:
#   1. The data path is relative to the repository root.
#   2. A seed is set. The 2018 notebook set none, so its exact OOB numbers
#      cannot be reproduced bit for bit; see REPEATS below for the spread.
#   3. Metrics are written to results/model_a_r.json.
#
# Usage (from the repository root):
#   Rscript R/reproduce_2018.R            # one fit, seed 2018
#   REPEATS=20 Rscript R/reproduce_2018.R # spread over 20 seeds

suppressPackageStartupMessages(library(randomForest))

load("data/ReefCheck974.rdata")  # creates `ReefCheck`
n_raw <- nrow(ReefCheck)

# ---- cleaning (verbatim logic from the 2018 notebook) ----------------------
recode <- function(x, blank_to, extra = NULL) {
  x <- as.character(x)
  x[x == ""] <- blank_to
  if (!is.null(extra)) for (from in names(extra)) x[x == from] <- extra[[from]]
  x
}
ReefCheck$Ocean       <- recode(ReefCheck$Ocean, "unknown")
ReefCheck$Storms      <- recode(ReefCheck$Storms, "unknown", c(y = "yes"))
ReefCheck$HumanImpact <- recode(ReefCheck$HumanImpact, "unknown")
ReefCheck$Siltation   <- recode(ReefCheck$Siltation, "never", c(Occasionally = "occasionally"))
ReefCheck$Dynamite    <- recode(ReefCheck$Dynamite, "none")
ReefCheck$Poison      <- recode(ReefCheck$Poison, "unknown")
ReefCheck$Sewage      <- recode(ReefCheck$Sewage, "none")
ReefCheck$Industrial  <- recode(ReefCheck$Industrial, "none")
ReefCheck$Commercial  <- recode(ReefCheck$Commercial, "none")

unknown_rows <- apply(ReefCheck, 1, function(r) any(r == "unknown"))
ReefCheck <- ReefCheck[!unknown_rows, ]

factor_cols <- c("Ocean", "Storms", "HumanImpact", "Siltation", "Sewage",
                 "Dynamite", "Poison", "Industrial", "Commercial")
for (col in factor_cols) ReefCheck[[col]] <- as.factor(ReefCheck[[col]])

# The notebook drops these with ReefCheck[-which(...), ]. That idiom deletes
# every row when which() is empty, so guard it; on this file both are non-empty.
drop_k <- which(ReefCheck$Sewage == "k")
if (length(drop_k)) ReefCheck <- ReefCheck[-drop_k, ]
drop_prior <- which(ReefCheck$Dynamite == "prior")
if (length(drop_prior)) ReefCheck <- ReefCheck[-drop_prior, ]

n_clean <- nrow(ReefCheck)
n_yes <- sum(ReefCheck$Bleaching == "Yes")
cat(sprintf("rows: raw %d, clean %d, bleaching Yes %d (%.2f%%)\n",
            n_raw, n_clean, n_yes, 100 * n_yes / n_clean))

# ---- model (verbatim call from the 2018 notebook) ---------------------------
fit_once <- function(seed) {
  set.seed(seed)
  fit <- randomForest(Bleaching ~ ., data = ReefCheck, ntree = 500, mtry = 5,
                      nodesize = sqrt(ncol(ReefCheck)), sampsize = c(300, 100),
                      importance = TRUE)
  cm <- fit$confusion[, 1:2]              # rows = observed, cols = predicted
  tn <- cm["No", "No"];  fp <- cm["No", "Yes"]
  fn <- cm["Yes", "No"]; tp <- cm["Yes", "Yes"]
  list(
    seed = seed,
    oob_error = unname(fit$err.rate[fit$ntree, "OOB"]),
    sensitivity = tp / (tp + fn),
    specificity = tn / (tn + fp),
    confusion = list(tn = tn, fp = fp, fn = fn, tp = tp),
    oob_votes_yes = unname(fit$votes[, "Yes"]),
    importance = fit$importance[, "MeanDecreaseAccuracy"]
  )
}

repeats <- as.integer(Sys.getenv("REPEATS", "1"))
seeds <- 2018 + seq_len(repeats) - 1
runs <- lapply(seeds, fit_once)
first <- runs[[1]]

cat(sprintf("seed %d: OOB error %.4f  sensitivity %.4f  specificity %.4f\n",
            first$seed, first$oob_error, first$sensitivity, first$specificity))
if (repeats > 1) {
  for (m in c("oob_error", "sensitivity", "specificity")) {
    v <- sapply(runs, `[[`, m)
    cat(sprintf("%-12s over %d seeds: mean %.4f  min %.4f  max %.4f\n",
                m, repeats, mean(v), min(v), max(v)))
  }
}

dir.create("results", showWarnings = FALSE)
json_num <- function(x) formatC(x, digits = 6, format = "fg", flag = "#")
spread <- function(m) {
  v <- sapply(runs, `[[`, m)
  sprintf('{"mean": %s, "min": %s, "max": %s}', json_num(mean(v)), json_num(min(v)), json_num(max(v)))
}
imp <- sort(first$importance, decreasing = TRUE)
out <- c(
  "{",
  sprintf('  "model": "A (2018 random forest, R randomForest %s)",', packageVersion("randomForest")),
  sprintf('  "rows_raw": %d, "rows_clean": %d, "positives": %d,', n_raw, n_clean, n_yes),
  sprintf('  "seed": %d, "oob_error": %s, "sensitivity": %s, "specificity": %s,',
          first$seed, json_num(first$oob_error), json_num(first$sensitivity), json_num(first$specificity)),
  sprintf('  "confusion": {"tn": %d, "fp": %d, "fn": %d, "tp": %d},',
          first$confusion$tn, first$confusion$fp, first$confusion$fn, first$confusion$tp),
  sprintf('  "repeats": %d,', repeats),
  sprintf('  "oob_error_over_seeds": %s,', spread("oob_error")),
  sprintf('  "sensitivity_over_seeds": %s,', spread("sensitivity")),
  sprintf('  "specificity_over_seeds": %s,', spread("specificity")),
  sprintf('  "importance_mean_decrease_accuracy": {%s}',
          paste(sprintf('"%s": %s', names(imp), json_num(imp)), collapse = ", ")),
  "}"
)
writeLines(out, "results/model_a_r.json")
write.csv(data.frame(oob_votes_yes = first$oob_votes_yes,
                     observed = as.character(ReefCheck$Bleaching)),
          "results/model_a_r_oob_votes.csv", row.names = FALSE)
cat("wrote results/model_a_r.json and results/model_a_r_oob_votes.csv\n")
