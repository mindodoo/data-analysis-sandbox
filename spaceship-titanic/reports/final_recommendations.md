# Final Recommendations — Agent 6

- Use tuned parameters for final training when ready to submit.
- Best OOF accuracy after tuning: **0.8108** (baseline was **0.8069**).
- Before generating `submission.csv`, retrain `best_pipeline` on full `train_clean` with group stats fit on all training passengers.
- Consider a threshold sweep on OOF probabilities if accuracy plateaus.
