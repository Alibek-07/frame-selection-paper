# ARKitScenes spike (2 evenings)

RESOLVED: Faro laser scans are ALREADY REGISTERED. Apply NO transform.
  _pose.txt = scanner tripod position in the registered frame, NOT a
  transform to apply. DATA.md is wrong on BOTH format and meaning:
  files are comma-delimited 4x4 (translation in last row), not
  whitespace 4x3.
  Evidence: identity gives 0.9cm median NN, 72.8% of points within 2cm;
  all 4 transform conventions give 30-49cm. Voxel collapse under
  identity 63% vs 31% for the best transform. Ceiling height agrees
  (2.36 vs 2.37m) either way.
  -> belongs in the paper's reproducibility appendix.

GT quality: sub-cm. Noise floor ~1cm.
Visit 484534: single studio, ~6x4m. floor z=-102.20, ceiling z=-99.83.

RETRACTED: earlier "ghosting", "8.9 deg yaw error", "inflated 19x19m
  envelope" were all artifacts of applying wrong transforms.

GT EXTRACTION: floor-footprint method works. Ceiling band gives best
  coverage (157k pts, irregularity 0.836) since nothing sits on a
  ceiling. Ceiling splits into 2 components (13.9 + 4.4 m2);
  largest-only truncates the room.
  KNOWN FIX: merge components >10% of largest.
  Floor OBB 6.34x3.02, ceiling(largest) 4.56x3.64; both imply ~18-19 m2.

UNACCOUNTED COST: running the arbuz pipeline on ARKitScenes video at
  all (format, resolution, ARKit poses). Larger than GT extraction.

DECISION: ARKitScenes parked. Main results on Tensor data first.
  Revisit only if a reviewer objection makes it necessary.
