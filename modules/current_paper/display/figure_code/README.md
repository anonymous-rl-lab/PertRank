# Submission figure reproduction

The current module entry point renders six main figures and seven supplementary figures from the included numerical summaries. `draw_main_figures.py` renders main1–6 and Supplementary6–7; `plot_supplement.py` renders Supplementary1–4; `plot_design_attenuation.py` renders Supplementary5. All write to `../figures/` relative to this display tree. These are display operations, separate from the four-arm and guide numerical replay.

Use the module entry point so all outputs remain outside the delivered repository. Source tables and plotting inputs retain all scientific estimates and uncertainty definitions.
