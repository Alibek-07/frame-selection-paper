# Sharper Frames, Worse Rooms

**Viewpoint coverage dominates image quality in budgeted frame selection for 3D reconstruction.**

When you can only keep a fixed number of frames from a video capture, the intuitive move is to keep the sharpest ones. This work tests that assumption across 324 configurations on nine capture sets and finds it is wrong. Where the frames were taken from matters more than how sharp they are, and selecting for image quality actively hurts reconstruction when the sharp frames cluster in one part of a scene.

Submitted to the NeurIPS 2026 Workshop on Vision-Language Models for Real-World Deployment.

## The question

Video based 3D reconstruction has to choose which frames to use. Processing every frame is too expensive, so some selection strategy runs first, and the standard assumption is that per frame image quality is the thing to optimize. Sharper frames, better reconstruction.

That assumption had not been tested directly. This work asks which property of a frame set actually predicts downstream reconstruction quality under a fixed frame budget, and measures it rather than reasoning about it.

## Approach

A full factorial design over 324 configurations, run across nine capture sets.

Frame selection strategies were varied against frame budgets, with reconstruction quality measured at the end of the pipeline rather than judged by eye. The design was factorial so that the effect of selection strategy could be separated from the effect of budget, instead of confounding the two.

Evaluation used quantitative reconstruction metrics together with vision language model confidence, so the result did not depend on a single measure that could be gamed by one strategy.

## What the experiment showed

Viewpoint coverage dominated per frame sharpness as a predictor of reconstruction quality.

The failure mode is specific and worth stating. Sharp frames tend to cluster, because a camera held steady produces several good frames of the same region. A selector optimizing for sharpness therefore spends its budget on one wall of a room and leaves the rest of the scene thin, producing a reconstruction that looks excellent in one direction and has holes everywhere else.

Selecting for coverage under the same budget produced better reconstructions even though the individual frames were worse.

## Why this matters

Frame selection is a preprocessing step that usually gets a one line heuristic and no further thought. The result says that heuristic is pointed at the wrong variable, and that the cost of getting it wrong is not uniform degradation but a specific and recoverable failure.

## Repository contents

```
scripts/        frame selection, the sweep harness, and evaluation
data_all/       capture sets used in the experiments
results/        sweep outputs and the figures in the paper
notes/          experiment notes and analysis
cache/          intermediate reconstruction artifacts
map-anything/   MapAnything, included as a submodule
paper/          the submitted paper
```

MapAnything is a submodule, so clone with submodules or initialize them after cloning.

## Running the experiments

```bash
git clone --recurse-submodules https://github.com/Alibek-07/REPO_NAME.git
cd REPO_NAME
pip install -r requirements.txt
```

TODO: add the actual sweep command from scripts/ here, plus any environment requirements such as GPU, CUDA version, or Python version.

## Data

TODO: say whether the capture sets in data_all can be redistributed. If they came from work you do not own, keep them out of the public repository and describe the capture format instead, clearly enough that someone could reproduce the experiment on their own captures.

## Citation

TODO: add the BibTeX entry once the paper has an arXiv identifier.

## Authors

Alibek Dadajonov and Hikmatillo Umarhojiyev.

## License

TODO: add a license file. MIT is the usual choice for work like this.
