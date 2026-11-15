# Model Card — Binary Classifier

## Latest Retraining Summary

| Metric          | Value                                              |
|-----------------|-----------------------------------------------------|
| Champion F1     | 0.5197537728926529            |
| Challenger F1   | 0.6716994146502163           |
| Trigger Reason  | scheduled_weekly           |
| Timestamp       | $(date -u +"%Y-%m-%d %H:%M:%S UTC")                 |
| Promoted        | Yes                                                  |

## Decision

The challenger model exceeded the champion by the required margin (>0.01 F1)
and passed the McNemar statistical significance test (p < 0.05).
The new model has been promoted to production.
