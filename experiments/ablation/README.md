# Matched Full-Population Reduced-Duration Ablation

The tracked A0-A8 configurations define the planned mechanism study. Every
condition uses the complete population stream, the complete L1/L2 expert union,
the same one-tenth-duration schedule and evaluation protocol. Confirmatory
conditions use seeds 13, 37, and 71. A1/A3 mask L1 labels from the objective,
but the same L1 tiles remain in the replay stream so image distribution and
replay frequency stay matched.

The prespecified contrasts are:

- A1 versus A3: multi-teacher contribution without the global prototype
  coordinate or L1 supervision, with matched L2 supervision retained;
- A2 versus A1: contribution of the global expert prototype coordinate,
  including direct L1, teacher-space semantic, and prototype-response
  supervision;
- A4 versus A3: expert prototype contribution in a single-teacher background;
- A0 versus A4: multi-teacher contribution with prototype supervision;
- A0 versus A2: per-tile teacher adjudication at the deployed filter strength;
- A0 versus A5: dynamic global prototype refresh;
- A0 versus A6: dynamic spatial prototype refresh;
- A0 versus A8: feedback of L2 gradients into the shared encoder.

A7 versus A0 is a filter-strength sensitivity analysis and is not interpreted
as a separate mechanism.

Every reported value is produced from the current spatial implementation and
its frozen run manifest. Generated results live in external experiment storage.

The tracked base configuration is an open-source example and contains
placeholder paths. Production runs supply the resolved local base through
`HCC_SEMPATH_ABLATION_BASE_CONFIG`; the runner overlays only the named
reduced-duration condition, retains local asset paths, executes in the
`hcc-camoe` conda environment, and removes its temporary resolved configs on
exit. With no condition arguments it runs A0-A8 across all three seeds:

```bash
HCC_SEMPATH_ABLATION_BASE_CONFIG=configs/local/server/train_full.yaml \
  bash experiments/ablation/scripts/run_ablations.sh
HCC_SEMPATH_ABLATION_BASE_CONFIG=configs/local/server/train_full.yaml \
  bash experiments/ablation/scripts/run_ablations.sh a0 a2 a4
HCC_SEMPATH_ABLATION_BASE_CONFIG=configs/local/server/train_full.yaml \
HCC_SEMPATH_ABLATION_SEEDS="13" \
  bash experiments/ablation/scripts/run_ablations.sh a0 a8
```
