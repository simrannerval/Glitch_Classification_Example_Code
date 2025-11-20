Example code for the glitch classification pipeline outlined in [Nerval et al. (2025, 2503.10798)](https://arxiv.org/abs/2503.10798) for TODs with injected simulated transient sources.

- ACT_DR6_detector_glitch_classification.ipynb: Example notebook explianing more about how the cuts pipeline works, this can also be found [here](https://github.com/ACTCollaboration/DR6_Notebooks/blob/main/ACT_DR6_detector_glitch_classification.ipynb).
- filter_stats_functions.py: Functions used to compute the summary statistics per glitch.
- sims_compute_filtering_values.py: Code used to compute the summary stats and make a dataframe with the glitches.
- Training_and_classification_functions.py: Functions used to train the random forest and classify glitches.
- Forest_and_classify.py: Code used to classify glitches.
- making_cuts_objects.py: Code used to make modified cuts objects for mapmaking.
- depth_1_for_sims.py: This code is written by Sigurd Naess with only minor modifications by Simran Nerval to work with sims. It is only included in the repo for completeness as it is needed to run the pipeline.
- run_full_pipeline_forsims.sh: Example script for how to run the whole glitch detection, glitch classification, and depth-1 (or single scan) mapmaking pipeline.
