# Artifacts hosted outside the repository

Two archives attached to the repository's [v1.0 release](https://github.com/zypapers/mamba-conductance-erf/releases/tag/v1.0), 2.35 GB and 80 files in all. `scripts/fetch_artifacts.sh` downloads both, checks their hashes and every file's hash in `ARTIFACTS.sha256`, and unpacks them into `checkpoints/` and `results/`.

| archive | bytes | sha256 |
|---|---|---|
| `mamba-conductance-erf-checkpoints.tar` | 1,353,492,480 | `95c4de64fa06ab19c469007bf660cc2367022f36df1a77d4bb83ded89d0e7ce0` |
| `mamba-conductance-erf-curves.tar` | 999,321,600 | `855a91e1180d1cd67b71a53fa2b2f2902203342f92b3d3e09b703275578ac832` |

## `mamba-conductance-erf-checkpoints.tar`

| file | bytes |
|---|---|
| `checkpoints/matched/seed1/final_default.pt` | 54,202,344 |
| `checkpoints/matched/seed1/final_fast.pt` | 54,202,293 |
| `checkpoints/matched/seed1/final_slow.pt` | 54,202,293 |
| `checkpoints/matched/seed2/final_default.pt` | 54,202,344 |
| `checkpoints/matched/seed2/final_fast.pt` | 54,202,293 |
| `checkpoints/matched/seed2/final_slow.pt` | 54,202,293 |
| `checkpoints/matched/seed3/final_default.pt` | 54,202,344 |
| `checkpoints/matched/seed3/final_fast.pt` | 54,202,293 |
| `checkpoints/matched/seed3/final_slow.pt` | 54,202,293 |
| `checkpoints/matched_nowd_timescale/seed1/final_default.pt` | 54,202,408 |
| `checkpoints/matched_nowd_timescale/seed1/final_fast.pt` | 54,202,357 |
| `checkpoints/matched_nowd_timescale/seed1/final_slow.pt` | 54,202,357 |
| `checkpoints/matched_nowd_timescale/seed2/final_default.pt` | 54,202,408 |
| `checkpoints/matched_nowd_timescale/seed2/final_fast.pt` | 54,202,357 |
| `checkpoints/matched_nowd_timescale/seed2/final_slow.pt` | 54,202,357 |
| `checkpoints/matched_nowd_timescale/seed3/final_default.pt` | 54,202,408 |
| `checkpoints/matched_nowd_timescale/seed3/final_fast.pt` | 54,202,357 |
| `checkpoints/matched_nowd_timescale/seed3/final_slow.pt` | 54,202,357 |
| `checkpoints/mqar/rerun_seed42/mqar_default.pt` | 2,776,279 |
| `checkpoints/mqar/rerun_seed42/mqar_fast.pt` | 2,776,228 |
| `checkpoints/mqar/rerun_seed42/mqar_slow.pt` | 2,776,228 |
| `checkpoints/mqar/rerun_seed43/mqar_default.pt` | 2,776,279 |
| `checkpoints/mqar/rerun_seed43/mqar_fast.pt` | 2,776,228 |
| `checkpoints/mqar/rerun_seed43/mqar_slow.pt` | 2,776,228 |
| `checkpoints/mqar/rerun_seed44/mqar_default.pt` | 2,776,279 |
| `checkpoints/mqar/rerun_seed44/mqar_fast.pt` | 2,776,228 |
| `checkpoints/mqar/rerun_seed44/mqar_slow.pt` | 2,776,228 |
| `checkpoints/selcopy/rerun_seed42/selcopy_default.pt` | 2,759,306 |
| `checkpoints/selcopy/rerun_seed42/selcopy_fast.pt` | 2,759,255 |
| `checkpoints/selcopy/rerun_seed42/selcopy_slow.pt` | 2,759,255 |
| `checkpoints/selcopy/rerun_seed43/selcopy_default.pt` | 2,759,306 |
| `checkpoints/selcopy/rerun_seed43/selcopy_fast.pt` | 2,759,255 |
| `checkpoints/selcopy/rerun_seed43/selcopy_slow.pt` | 2,759,255 |
| `checkpoints/selcopy/rerun_seed44/selcopy_default.pt` | 2,759,306 |
| `checkpoints/selcopy/rerun_seed44/selcopy_fast.pt` | 2,759,255 |
| `checkpoints/selcopy/rerun_seed44/selcopy_slow.pt` | 2,759,255 |
| `checkpoints/model_2layer.pt` | 56,941,999 |
| `checkpoints/model_slow.pt` | 54,202,293 |
| `checkpoints/model_default.pt` | 162,606,133 |
| `checkpoints/model_fast.pt` | 54,202,293 |

## `mamba-conductance-erf-curves.tar`

| file | bytes |
|---|---|
| `results/camera_ready/matched/single_layer_default_seed1_test.npz` | 26,591,453 |
| `results/camera_ready/matched/single_layer_default_seed1_validation.npz` | 23,104,136 |
| `results/camera_ready/matched/single_layer_default_seed2_test.npz` | 26,302,709 |
| `results/camera_ready/matched/single_layer_default_seed2_validation.npz` | 22,832,869 |
| `results/camera_ready/matched/single_layer_default_seed3_test.npz` | 26,230,563 |
| `results/camera_ready/matched/single_layer_default_seed3_validation.npz` | 22,845,982 |
| `results/camera_ready/matched/single_layer_fast_seed1_test.npz` | 26,386,014 |
| `results/camera_ready/matched/single_layer_fast_seed1_validation.npz` | 22,941,319 |
| `results/camera_ready/matched/single_layer_fast_seed2_test.npz` | 26,324,846 |
| `results/camera_ready/matched/single_layer_fast_seed2_validation.npz` | 22,887,651 |
| `results/camera_ready/matched/single_layer_fast_seed3_test.npz` | 26,505,800 |
| `results/camera_ready/matched/single_layer_fast_seed3_validation.npz` | 23,082,370 |
| `results/camera_ready/matched/single_layer_slow_seed1_test.npz` | 26,520,403 |
| `results/camera_ready/matched/single_layer_slow_seed1_validation.npz` | 23,076,859 |
| `results/camera_ready/matched/single_layer_slow_seed2_test.npz` | 26,528,306 |
| `results/camera_ready/matched/single_layer_slow_seed2_validation.npz` | 23,080,626 |
| `results/camera_ready/matched/single_layer_slow_seed3_test.npz` | 26,549,523 |
| `results/camera_ready/matched/single_layer_slow_seed3_validation.npz` | 23,076,398 |
| `results/camera_ready/matched_nowd_timescale/single_layer_default_seed1_test.npz` | 26,651,456 |
| `results/camera_ready/matched_nowd_timescale/single_layer_default_seed1_validation.npz` | 23,190,883 |
| `results/camera_ready/matched_nowd_timescale/single_layer_default_seed2_test.npz` | 26,300,640 |
| `results/camera_ready/matched_nowd_timescale/single_layer_default_seed2_validation.npz` | 22,863,974 |
| `results/camera_ready/matched_nowd_timescale/single_layer_default_seed3_test.npz` | 26,149,352 |
| `results/camera_ready/matched_nowd_timescale/single_layer_default_seed3_validation.npz` | 22,728,532 |
| `results/camera_ready/matched_nowd_timescale/single_layer_fast_seed1_test.npz` | 26,474,241 |
| `results/camera_ready/matched_nowd_timescale/single_layer_fast_seed1_validation.npz` | 23,090,231 |
| `results/camera_ready/matched_nowd_timescale/single_layer_fast_seed2_test.npz` | 26,444,528 |
| `results/camera_ready/matched_nowd_timescale/single_layer_fast_seed2_validation.npz` | 22,963,031 |
| `results/camera_ready/matched_nowd_timescale/single_layer_fast_seed3_test.npz` | 26,284,039 |
| `results/camera_ready/matched_nowd_timescale/single_layer_fast_seed3_validation.npz` | 22,861,234 |
| `results/camera_ready/matched_nowd_timescale/single_layer_slow_seed1_test.npz` | 25,952,592 |
| `results/camera_ready/matched_nowd_timescale/single_layer_slow_seed1_validation.npz` | 22,609,666 |
| `results/camera_ready/matched_nowd_timescale/single_layer_slow_seed2_test.npz` | 26,065,948 |
| `results/camera_ready/matched_nowd_timescale/single_layer_slow_seed2_validation.npz` | 22,752,258 |
| `results/camera_ready/matched_nowd_timescale/single_layer_slow_seed3_test.npz` | 26,062,680 |
| `results/camera_ready/matched_nowd_timescale/single_layer_slow_seed3_validation.npz` | 22,716,376 |
| `results/task_sensitivity_v2_records.npz` | 3,276,311 |
| `results/task_probe_control_curves.npz` | 20,819,852 |
| `results/two_layer/scan_probe_large_curves.npz` | 7,741,380 |
| `results/pretrained/scan_probe_large_mamba-370m_curves.npz` | 80,375,378 |
