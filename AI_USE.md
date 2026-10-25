# AI Use Log

| Date | Tool | What I asked / used it for | What I kept / changed | How I checked it |
|------|------|----------------------------|-----------------------|------------------|
| 22/10 | Claude | Summary of brief, project plan, setup steps (GitHub, Git, folders) | Followed setup steps; chose dataset 3 myself | Checked against project brief |

| 22/10 | Claude | Data exploration script, polygon-label fix, Colab training setup, README draft | Ran and checked outputs; counts verified against Roboflow (1446/18) | Compared script counts with Roboflow class page |

# AI Use Log

Tool used: Claude (Anthropic), via chat, 22–23 Oct 2026.
Approximate AI contribution: **high** for code and planning; data choices,
running all experiments, checking outputs and final decisions were mine.

| Date | What I asked / AI did | What I kept / changed | How I checked it |
|------|----------------------|-----------------------|------------------|
| 22/10 | Summarised brief, proposed timeline and project scope | Chose dataset 3 myself; kept narrow scope | Compared against project brief and rubric |
| 22/10 | Step-by-step setup: GitHub, Git, venv, folder structure | Followed steps; fixed my own venv activation issue | Ran checks (`All good!`), pushed to GitHub |
| 22/10 | Suggested rebalancing Roboflow split (90/6/4 → 70/20/10) | I created dataset version 2 | Confirmed split counts in Roboflow and Colab (1369/391/195) |
| 22/10 | Wrote `explore.py` (counts, box sizes, label validation) | Kept; found 2 polygon labels | Counts matched Roboflow exactly (1446 / 18) |
| 22/10 | Colab training code (YOLOv8n), polygon-label fix | Restarted training after fix | Checked `0 corrupt` in training log |
| 22–23/10 | Explained YOLO, training, mAP, transfer learning | Asked follow-up questions until I understood | Confirmed use of pretrained library with teaching team on edStem |
| 23/10 | Planned alternatives (YOLOv8s) and resolution sweep (320/480/640) | Ran all trainings; chose YOLOv8n-480 based on results | Compared validation tables |
| 23/10 | Suggested trapezoid/Simpson for AP and golden-section for threshold; wrote `numerics.py`, `analysis.py`, `inspect_tray.py` and all tests | Kept methods after comparing with other Week 2/6 methods | 54 tests pass (incl. known exact integrals); my AP matched Ultralytics to 0.0003 |
| 23/10 | Evaluation cells: speed, FLOPs, FP16 round-off, final test | Ran once on test set; did not tune after seeing test results | Checked outputs were consistent with validation |
| 23/10 | README results section, pdoc docs, branch + pull request | Reviewed README content | Opened docs in browser; PR merged |
| 25/10 | ChatGPT: grammar, flow and organisation of the report; formatting references | Reviewed all suggested changes | Checked the meaning of my writing was unchanged |

## Errors / unhelpful AI suggestions
- Initial training cell referenced `RUNS` before it was defined (fixed by running Drive cell first).
- Colab session resets meant re-running setup cells; the AI's first instructions didn't make this clear.
- A copy of `numerics.py` ended up in the `tests/` folder by mistake; I found and removed it.
- A stray space in the Colab file path created a second `notebooks` folder; I moved the notebook and fixed it.

## Decisions that remained mine
- Choosing Dataset 3 (pork rashers) out of the available project datasets.
- Creating the rebalanced Roboflow dataset version (70/20/10) and checking the split counts.
- Working with VS Code for code/tests and Colab for GPU training, rather than doing everything in Colab.
- Running every training, evaluation and test myself, and checking each output before moving on.
- Choosing the final model and settings (YOLOv8n, 480 px, FP32, threshold 0.137) from the validation results.
- Choosing to stop tuning after the test set was used, so the test result stayed honest.
- Deciding to keep the wrinkle class and report its poor result honestly, rather than removing it.
- Deciding which results and figures to include in the report, and which to move to the appendix.
- Choosing the example test image used in the report and describing what it shows.
- Writing the final report in my own words.