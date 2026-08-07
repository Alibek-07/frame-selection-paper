## Primary metric
Two-part. NEVER collapsed into one number.

(a) SUCCESS RATE: fraction of captures where pipeline emits a
    usable floor plan at all
(b) DIMENSION ERROR | SUCCESS: mean absolute relative error on
    principal wall-to-wall distances (length, width), over
    successful captures only

## Error form
Relative % = primary. Absolute cm = secondary table.
Rationale: room scale varies; 20cm means different things in
a closet and a living room.

## Aggregation
Primary:  mean relative error
Tail:     fraction of successful rooms with >5% error
Also report: >3% and >10% for a sensitivity sweep
Rationale: if coverage's advantage is preventing blowups, the
mean hides it and the tail reveals it.

## Scale alignment
NONE. Metric depth = absolute scale claim. No global scale fit
before comparison. State explicitly in paper - reviewers ask.

## Budget unit
Primary:   absolute frame count K (compute cost is per-frame)
Secondary: K as fraction of sequence length
NOTE: production ONNX export is fixed at 16 views. K=16 is the
highlighted operating point inside the PyTorch sweep.

## Exclusions
Rule fixed in advance: no capture is dropped. Total failures
go in the success-rate table, not the trash.

## Why dimensions, not pointwise geometry
Wall-to-wall distances are scalars, so they need no coordinate
alignment between prediction and GT. A pointwise metric
(chamfer) would require Faro<->ARKit registration.

## Rooms are non-rectangular
Primary = OBB extents along the two dominant Manhattan axes.
Covariate = floor_area / OBB_area as an irregularity measure.
Open question: does coverage help more in irregular rooms?
