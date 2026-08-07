# Related Work — positioning table (v5)

Six works read in full: Ahmed 2010, Wang 2014, Zhang 2017, Koschel 2021,
Jha 2025, + submodular formalism. `[verify]` = genuinely unread.

## CONTRIBUTION CHANGE — read before drafting

DEAD: "we propose coverage-based frame selection." Wang et al. (ICME 2014)
optimise a GLOBAL viewing-angle coverage objective at FIXED cardinality.
Their comparison baseline divides 360 deg into n=36 groups, treating frames
within a 10-degree range as one viewing-angle bin, and picks the best frame
per bin. That is exactly our coverage_score (36 bins). Our strategy is their
baseline. Their TED method beats it by 41.93% on coverage gain difference.

THEREFORE: coverage is established prior art we TEST AGAINST QUALITY, cited
to Wang et al., and our greedy bin-coverage selector is explicitly identified
as their heuristic. State this in the method section, not buried.

ALIVE — the paper rests entirely on three claims:
1. MECHANISM — quality-ranked selection is HARMFUL at fixed budget, and the
   cause is loss of coverage. No prior work considers the conflict.
2. REGIME — feed-forward multi-view metric reconstruction (MapAnything);
   rotation-dominant, planar handheld interior capture = the classical
   degenerate case. No F estimated, no correspondence, no bundle adjustment.
3. TASK METRIC — room dimension error + success rate, vs reprojection error /
   Chamfer / user study.

## Three arguments that carry the section

1. ROOM SWEEPS ARE THE DEGENERATE CASE CLASSICAL SELECTION REJECTS.
Ahmed et al. (VISAPP 2010) disqualify motion degeneracy (rotation about the
camera centre -> epipolar geometry undefined) and structure degeneracy
(coplanar points -> F undetermined). A handheld sweep is rotation-dominant;
walls are planar. Koschel et al. (2021) make the same criticism
independently: the method "discards frames that can contain information of
interest and assumes a recording strategy without camera rotation about its
axis." Cite both.

2. THE QUALITY/COVERAGE TENSION IS KNOWN BUT NEVER TESTED.
- Koschel Fig. 1 contrasts good-quality-but-static vs varied-viewpoint-but-
  blurred, resolves toward quality, never tests the alternative, lists
  overlap-based selection as future work.
- Zhang et al. structurally avoid the conflict: geometric spacing is a hard
  constraint, blur a local refinement inside it.
- Wang et al. optimise coverage alone and never consider quality.
No one has run the coverage-matched control.

3. PRIOR SETTINGS HAVE COVERAGE ALREADY GUARANTEED — OURS DOES NOT.
- Zhang 2017 (drone nadir, gimbal, constant altitude/velocity, planned
  S-path): reprojection error 0.473/0.478/0.468/0.477/0.472 vs uniform
  0.475/0.425/0.421/0.436/0.476 — uniform competitive or better in most sets.
- Wang 2014 (crowdsourced multi-user, outdoor, object-centric): user study
  scores mostly near 2 = indistinguishable from 1-second fixed-duration
  sampling; worse on objects 4 and 8.
- Koschel 2021 (360 rig + IMU): RMSE 1.041 -> 0.980, attributed to Meshroom's
  outlier detection already absorbing bad frames.
In each, coverage is enforced by the rig, the flight plan, or the multiplicity
of contributors. A single handheld interior sweep with order-of-magnitude
variation in angular velocity is where coverage is genuinely at risk.
TESTABLE: report angular-velocity variance for our captures.

## Family A — classical, triangulation-based  [READ]

Ahmed, M.T.; Dailey, M.N.; Landabaso, J.L.; Herrero, N. "Robust Key Frame
Extraction for 3D Reconstruction from Video Streams." VISAPP 2010, 231-236.
Confirmed 3x (Koschel [5], Zhang [9], Wang [6]).

Correspondence ratio ~ baseline proxy; Torr GRIC (homography vs F);
point-to-epipolar-line cost; REJECTS rotation + coplanar structure;
metric = root-mean reprojection error; budget variable (29-84 frames).
Own result: GRIC lowers reprojection VARIANCE but not MEAN vs uniform.

Predecessor chain (cross-confirmed):
- Seo, J.K.; Kim, S.H.; Jho, C.W.; Hong, H.K. ITC-CSCC 2003, 1282-1285.
- Seo, Y.-H.; Kim, S.-H.; Doo, K.-S.; Choi, J.-S. Optical Engineering 2008,
  47(5), 525-534.
- Torr, P.H.S. "Geometric Motion Segmentation and Model Selection."
  Phil. Trans. R. Soc. A, 1998. (Wang [9]; Zhang [14] cites an ICCV 1998
  Torr/Fitzgibbon/Zisserman paper — resolve which we need)
- Pollefeys et al. IJCV 2004, 59(3), 207-232 (Ahmed cites a JVCA 2002
  version — resolve).
- Park, M.G.; Yoon, K.J. Electron. Lett. 2011, 47(25), 1367-1369.
- Mordohai et al. "Real-time Video-based Reconstruction of Urban
  Environments." ISPRS 2007. [verify] Per Wang: selects frames whose baseline
  exceeds a threshold, and notes the threshold VARIES WITH SCENE DEPTH.

## Family B — quality / blur  <- the assumption we contradict

Koschel, A.; Muller, C.; Reiterer, A. "Selection of Key Frames for 3D
Reconstruction in Real Time." Algorithms 2021, 14, 303.
doi:10.3390/a14110303 (open access)  [READ]

VuzeXR 360 + IMU; 2D-DFT magnitude on 72x128 boxes (blur threshold 98.4);
Lucas-Kanade 0.83 px to drop STATIC frames; random forest on IMU features;
Meshroom SfM; RMSE of reprojection residuals; budget 1535 -> 474;
NO coverage-matched control.
Table 2: RMSE 1.041 -> 0.980; time 1163.54s -> 169.835s.
Classifier: bad P .905/R .918; good P .695/R .661 ("slightly better than a
random guess"); conclusions call performance wrt ideal frames "unsatisfying".
Acknowledged conflict: blur rejection wants slow motion, redundancy rejection
wants motion. FUTURE WORK = our contribution (overlap thresholds).

Others in Family B:
- Ahmed PELC term [read] — proxy metric, variable budget
- Zhang blur stage [read] — Crete metric as LOCAL refinement after spacing
- Crete, F.; Dolmiere, T.; Ladret, P. "The blur effect: perception and
  estimation with a new no-reference perceptual blur metric." SPIE
  Electronic Imaging 2007, 64920I. <- the standard blur metric here
- Rashidi, A.; Dai, F.; Brilakis, I. et al. ASCE ICCCE 2012, p.188 [verify]
- CVPR 2023 UG2+ turbulence, arXiv:2306.08963 — distorted frames make a
  "negative contribution"; selects the sharpest set
- Lucky-frame / lucky imaging [verify, via arXiv:1712.03825]
- Ishijima et al. J. Biomed. Opt. 2015, 20, 46014 [verify] — per Koschel,
  MINIMISES frame-to-frame variation, i.e. deliberately keeps redundancy
- Ren, Shen, Lin, Mech. WACV 2020, 3201-3210 [verify] — CNN picks ONE frame

## Family C — global coverage objectives  <- OUR ALGORITHM LIVES HERE

Wang, G.; Lu, Y.; Zhang, L.; Alfarrarjeh, A.; Zimmermann, R.; Kim, S.H.;
Shahabi, C. "Active Key Frame Selection for 3D Model Reconstruction from
Crowdsourced Geo-tagged Videos." ICME 2014, 1-6. NUS + USC IMSC.  [READ]

- Coverage objective: GLOBAL. Minimise average expected square coverage gain
  difference over all frames. Derived to depend only on the selected set.
- Formalism: Transductive Experimental Design; NP-hard; convex relaxation
  (Yu et al.); Manifold Adaptive Kernel in RKHS.
- Cardinality: FIXED (t held constant across methods).
- Their baseline: 360 deg -> n=36 groups, 10-deg bins, best frame per group.
  IDENTICAL TO OUR METHOD.
- Quality/blur: NEVER CONSIDERED. Purely GPS + compass metadata.
- Capture: 345 UGVs, 77,642 frames, many users, OUTDOOR, OBJECT-CENTRIC
  (on a circle looking in). Ours: one handheld interior sweep (looking out).
- Pipeline: SIFT/VLFeat -> bundler -> CMVS -> PMVS2 (triangulation).
- Ground truth: NONE. "we currently do not have a groundtruth 3D model";
  22-person user study instead.
- Result vs naive: scores mostly ~2 = indistinguishable from 1s fixed-duration.

Formalism trail from their refs: Yu, Bi & Tresp "Transductive Experiment
Design" 2005 / ICML 2006; Yu, Zhu, Xu & Gong SIGIR 2008 (convex TED);
Cai & He "Manifold Adaptive Experimental Design" IEEE TKDE 2012;
Sindhwani, Niyogi & Belkin ICML 2005.

Also Family C:
- Zhang, C.; Wang, H.; Li, H.; Liu, J. "A Fast Key Frame Extraction Algorithm
  and an Accurate Feature Matching Method for 3D Reconstruction from Aerial
  Video." CCDC 2017, Chongqing, 6744-6749. Nankai Univ.  [READ]
  Overlap is a PAIRWISE SPACING RULE:
  L = 2(H - dH + dh) * tan(theta + dtheta) * (1 - sigma_min), from altitude
  and FOV. Blur (Crete) checked AFTER position is fixed, searching nearby
  frames. Then GRIC + correspondence ratio + N_min. VisualSFM + PMVS/CMVS.
  Stated contribution: selection SPEED (43s vs 752s).
- Xie, Wan, Bu, Zhou. "Aerial Sequential Frame Decimation for Scene
  Reconstruction." IEEE ICIA 2015, 1377-1381. [verify]
- Towards Scalable Multi-View Reconstruction, arXiv:2306.03747 — removes view
  nearest its neighbour ~ FPS; implementation detail there.

## Family D — learned / adaptive  [READ]

Jha, R.; Zhou, Y.; Loianno, G. "Adaptive Keyframe Selection for Scalable 3D
Scene Reconstruction in Dynamic Environments." arXiv:2510.23928v3,
28 Dec 2025. NYU / UC Berkeley.

RGB-D (stated limitation: "assumes the availability of depth data for the
warping module"); photometric L1 (0.7) + SSIM (0.3) on depth-warped last
keyframe; THRESHOLD-based, KFCR an outcome; Spann3r / CUT3R backbones;
Acc/Comp/Chamfer/NC metrics; baselines static, inertial [Piao & Kim, IEEE TMM
21(11) 2019], optimisation [Chakraborty, Tickoo, Iyer, WACV 2015];
NO coverage baseline.
NOTE: same feed-forward multi-view family as MapAnything — our Family D
differentiation is RGB-D vs RGB, threshold vs fixed-K, geometric vs task
metric, NOT "different model class".
Handle honestly: their signal is temporally local redundancy avoidance, which
correlates with coverage; they note the system is "most responsive when scene
coverage is at risk."
Quotable concessions: uniform sampling can be very effective in regular
texture-rich scenes; the primary win is efficiency rather than quality.

## Family E — the task
360-DFPE arXiv:2112.06180; DeepPerimeter arXiv:1904.11595;
PixCuboid arXiv:2508.04659; PlanarRecon, UniPlane [verify];
Rent3D CVPR 2015. None study frame selection.

## Family F — blur-tolerant reconstruction
Remove or model, never keep for coverage. Deblur-NeRF, PDRF, DP-NeRF,
BAD-NeRF [verify via arXiv:2403.19780]; arXiv:2403.19780;
arXiv:1903.06531 (blurred frames encode relative motion).

## Family G — video-LLM frame selection
MDP3, arXiv:2501.02885 — frame selection as monotone submodular maximisation
under cardinality constraint, (1-1/e) greedy proof. TAKEN; cite, don't claim.
Tang et al. "Adaptive Keyframe Sampling for Long Video Understanding."
CVPR 2025, 29118-29128. [verify]

## Our regime: feed-forward multi-view (NEW - MapAnything)

Keetha, N. et al. "MapAnything: Universal Feed-Forward Metric 3D
Reconstruction." arXiv:2509.13414. Meta + CMU.
- Transformer ingests N images jointly, regresses metric geometry + cameras.
- No F, no correspondence, no bundle adjustment -> classical degeneracy
  conditions do not apply. Argument 1 SURVIVES.
- But it IS multi-view: "baseline is irrelevant" is NOT assumable. Whether
  classical view-geometry criteria transfer is an OPEN QUESTION we test.
  mean_baseline is the control.
- Order-sensitive (implicit coordinate anchoring to a reference view).
  MUST fix view order across strategies. Pi3-X is permutation-equivariant ->
  natural pairing for the ablation.
- Production serves a 16-view ONNX export; fewer views are PADDED by
  repeating the last frame, so K != 16 must use PyTorch. Report parity at 16.
- use_multiview_confidence gives cross-view depth agreement directly = the
  middle link of our mechanism, measured not asserted.

## Formalism (rigour, not novelty)

Two framings, both prior art:
- Submodular / coverage maximisation: NWF 1978, tight (1-1/e) (Feige 1998).
  Frame-selection precedent: MDP3.
- Experimental design (TED): Wang et al. 2014; Yu, Bi & Tresp.
Position honestly: stronger coverage optimisers exist; we deliberately use
the simple greedy bin-coverage selector — THEIR heuristic baseline — because
our object of study is the trade-off, not the optimiser.

## Positioning paragraph (draft)

Frame selection for 3D reconstruction has been studied for two decades.
Classical criteria maximise triangulation baseline and reject motion- and
structure-degenerate frames [Ahmed 2010; Pollefeys 2004; Torr 1998] — yet a
handheld room sweep is dominated by rotation about the operator and by planar
walls, exactly what those methods discard, as Koschel et al. [2021] note.
Quality-based criteria reject motion blur [Ahmed 2010; Rashidi 2012;
Koschel 2021], validated against reprojection error rather than a task metric
and never at fixed budget. Coverage-based selection is well established:
Wang et al. [2014] optimise a global viewing-angle coverage objective at fixed
cardinality via transductive experimental design, and the angular-bin selector
we adopt is their comparison heuristic. Where coverage and quality co-occur
they do not compete — Zhang et al. [2017] fix geometric spacing first and
treat blur as a local refinement within it, while Wang et al. consider no
image quality at all.

Notably, these settings guarantee coverage by construction: gimbal-stabilised
nadir flight at constant velocity [Zhang 2017], a rig-mounted 360 camera
[Koschel 2021], or many independent contributors circling a landmark
[Wang 2014]. Consistent with this, each reports coverage-aware selection
performing close to uniform or fixed-duration sampling. A single handheld
interior sweep is different: angular velocity varies by an order of magnitude,
so temporal uniformity decouples from angular coverage.

We study that regime, with metric geometry predicted by a feed-forward
multi-view model and evaluation against a task metric (room dimension error
and reconstruction success rate). Our contribution is not the selection
algorithm, which is prior art, but the finding that the inherited quality
heuristic is actively harmful here, together with a 2x2 factorial isolating
viewpoint spread from per-frame quality as the causal factor.

## TODO
- [ ] measure angular-velocity variance in our captures (argument 3 -> data)
- [ ] resolve Torr 1998: Phil. Trans. R. Soc. A vs ICCV
- [ ] resolve Pollefeys: JVCA 2002 vs IJCV 2004
- [ ] [verify]: Rashidi 2012, Xie 2015, Ishijima 2015, Ren 2020,
      thermal-video paper, Mordohai 2007
- [ ] refs identified ~40; trim to 18-25 for a workshop paper
