## Task 4 — Similarity Threshold Calibration

The threshold was calibrated using 5 in-scope queries and 4
out-of-scope queries against the fixed-size ChromaDB collection.

[table]

### Observed similarity ranges

- In-scope: 0.4859 – 0.7930
- Out-of-scope: -0.0160 – 0.0733

### Selected threshold

Threshold: 0.30

The threshold was selected after observing the calibration results.
The highest observed out-of-scope similarity was 0.0733, while the
lowest observed in-scope similarity was 0.4859. A threshold of 0.30
lies between these observed clusters and provides separation between
the two groups.