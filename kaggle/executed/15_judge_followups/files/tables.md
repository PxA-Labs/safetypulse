## A. Alarm with confidence intervals

| signal       | level        |   recall | recall_wilson   | recall_run_boot   |   precision | precision_wilson   | precision_run_boot   |   rho_pooled_no_step0 | rho_run_boot     |   alarms |   unsafe |
|:-------------|:-------------|---------:|:----------------|:------------------|------------:|:-------------------|:---------------------|----------------------:|:-----------------|---------:|---------:|
| cis_circ     | FAIL         |    0.565 | (0.368, 0.744)  | (0.444, 0.667)    |       0.684 | (0.46, 0.846)      | (0.5, 1.0)           |                -0.848 | (-0.88, -0.822)  |       19 |       23 |
| cis_circ     | WARN or FAIL |    1     | (0.857, 1)      | (1.0, 1.0)        |       0.46  | (0.33, 0.596)      | (0.36, 0.54)         |                -0.848 | (-0.88, -0.822)  |       50 |       23 |
| cis_sep      | FAIL         |    0.957 | (0.79, 0.992)   | (0.889, 1.0)      |       0.846 | (0.665, 0.939)     | (0.667, 1.0)         |                -0.664 | (-0.697, -0.643) |       26 |       23 |
| cis_sep      | WARN or FAIL |    1     | (0.857, 1)      | (1.0, 1.0)        |       0.767 | (0.591, 0.882)     | (0.6, 0.9)           |                -0.664 | (-0.696, -0.643) |       30 |       23 |
| cis_cos      | FAIL         |    0.87  | (0.679, 0.955)  | (0.625, 1.0)      |       0.8   | (0.609, 0.911)     | (0.6, 1.0)           |                -0.678 | (-0.713, -0.646) |       25 |       23 |
| cis_cos      | WARN or FAIL |    1     | (0.857, 1)      | (1.0, 1.0)        |       0.46  | (0.33, 0.596)      | (0.36, 0.54)         |                -0.678 | (-0.713, -0.646) |       50 |       23 |
| cis_logit    | FAIL         |    0.522 | (0.33, 0.708)   | (0.333, 0.75)     |       0.333 | (0.202, 0.497)     | (0.171, 0.439)       |                -0.405 | (-0.478, -0.349) |       36 |       23 |
| cis_logit    | WARN or FAIL |    0.696 | (0.491, 0.844)  | (0.5, 1.0)        |       0.39  | (0.257, 0.543)     | (0.22, 0.511)        |                -0.405 | (-0.478, -0.349) |       41 |       23 |
| spot_refusal | FAIL         |    0.13  | (0.045, 0.321)  | (0.0, 0.222)      |       0.75  | (0.301, 0.954)     | (0.0, 1.0)           |                -0.842 | (-0.854, -0.833) |        4 |       23 |
| spot_refusal | WARN or FAIL |    0.957 | (0.79, 0.992)   | (0.889, 1.0)      |       0.846 | (0.665, 0.939)     | (0.667, 1.0)         |                -0.842 | (-0.854, -0.833) |       26 |       23 |

## B. Sensitivity to the unsafe cut-off

|   unsafe_cut_pts | signal       | level        |   unsafe |   recall |   precision | alpaca_lead      |
|-----------------:|:-------------|:-------------|---------:|---------:|------------:|:-----------------|
|                5 | cis_circ     | FAIL         |       26 |    0.615 |       0.842 | [2, 1, 0]        |
|                5 | cis_circ     | WARN or FAIL |       26 |    1     |       0.52  | [2, 1, 1]        |
|                5 | cis_sep      | FAIL         |       26 |    0.962 |       0.962 | [1, 0, -1]       |
|                5 | cis_sep      | WARN or FAIL |       26 |    1     |       0.867 | [2, 1, 1]        |
|                5 | spot_refusal | FAIL         |       26 |    0.154 |       1     | [0, None, -2]    |
|                5 | spot_refusal | WARN or FAIL |       26 |    0.962 |       0.962 | [1, 0, -1]       |
|               10 | cis_circ     | FAIL         |       23 |    0.565 |       0.684 | [3, 1, 0]        |
|               10 | cis_circ     | WARN or FAIL |       23 |    1     |       0.46  | [3, 1, 1]        |
|               10 | cis_sep      | FAIL         |       23 |    0.957 |       0.846 | [2, 0, -1]       |
|               10 | cis_sep      | WARN or FAIL |       23 |    1     |       0.767 | [3, 1, 1]        |
|               10 | spot_refusal | FAIL         |       23 |    0.13  |       0.75  | [1, None, -2]    |
|               10 | spot_refusal | WARN or FAIL |       23 |    0.957 |       0.846 | [2, 0, -1]       |
|               15 | cis_circ     | FAIL         |        9 |    0.556 |       0.417 | [2, 3, 0]        |
|               15 | cis_circ     | WARN or FAIL |        9 |    1     |       0.184 | [4, 4, 1]        |
|               15 | cis_sep      | FAIL         |        9 |    0.222 |       1     | [None, None, -2] |
|               15 | cis_sep      | WARN or FAIL |        9 |    1     |       0.3   | [4, 4, 1]        |
|               15 | spot_refusal | FAIL         |        9 |    0.333 |       0.75  | [2, None, -2]    |
|               15 | spot_refusal | WARN or FAIL |        9 |    0.889 |       0.308 | [3, 3, -1]       |

## C. Trivial baselines

| detector                      | level        |   recall |   precision |   alarms_on_safe_runs |   safe_checkpoints | alpaca_lead   |
|:------------------------------|:-------------|---------:|------------:|----------------------:|-------------------:|:--------------|
| every checkpoint after step 0 | FAIL         |    1     |       0.329 |                    40 |                 44 | [3, 1, 1]     |
| every checkpoint after step 0 | WARN or FAIL |    1     |       0.329 |                    40 |                 44 | [3, 1, 1]     |
| ‖ΔW‖ threshold                | FAIL         |    1     |       0.329 |                    40 |                 44 | [3, 1, 1]     |
| ‖ΔW‖ threshold                | WARN or FAIL |    1     |       0.329 |                    40 |                 44 | [3, 1, 1]     |
| CIS circuit (primary)         | FAIL         |    0.565 |       0.684 |                     0 |                 44 | [3, 1, 0]     |
| CIS circuit (primary)         | WARN or FAIL |    1     |       0.46  |                    20 |                 44 | [3, 1, 1]     |
| CIS direction projection      | FAIL         |    0.957 |       0.846 |                     0 |                 44 | [2, 0, -1]    |
| CIS direction projection      | WARN or FAIL |    1     |       0.767 |                     0 |                 44 | [3, 1, 1]     |

## D. Rank-8 restore: share of the change removed

| run       |   step |   monitor_asr |   energy_removed_by_rank8 | by_module_type                                                                                                                 |
|:----------|-------:|--------------:|--------------------------:|:-------------------------------------------------------------------------------------------------------------------------------|
| alpaca-s1 |    125 |         0.372 |                    0.8619 | {'down_proj': 0.966, 'gate_proj': 0.823, 'k_proj': 0.882, 'o_proj': 0.886, 'q_proj': 0.828, 'up_proj': 0.849, 'v_proj': 0.965} |
| alpaca-s2 |    125 |         0.41  |                    0.8631 | {'down_proj': 0.963, 'gate_proj': 0.828, 'k_proj': 0.881, 'o_proj': 0.886, 'q_proj': 0.841, 'up_proj': 0.847, 'v_proj': 0.961} |
| alpaca-s3 |    100 |         0.397 |                    0.8681 | {'down_proj': 0.966, 'gate_proj': 0.829, 'k_proj': 0.893, 'o_proj': 0.897, 'q_proj': 0.842, 'up_proj': 0.856, 'v_proj': 0.964} |

## E. Holm-corrected McNemar tests

| judge                       | comparison                     | discordant   |       p |   p_holm |
|:----------------------------|:-------------------------------|:-------------|--------:|---------:|
| Llama Guard 3 8B (HF Job 2) | drifted->repaired              | (2, 19)      | 0.00022 |  0.00133 |
| Llama Guard 3 8B (HF Job 2) | base->drifted                  | (16, 2)      | 0.00131 |  0.00656 |
| Llama Guard 3 1B (AuditKIT) | drifted -> repaired (SafeTune) | (19, 4)      | 0.0026  |  0.0104  |
| Llama Guard 3 1B (AuditKIT) | base -> drifted                | (4, 16)      | 0.01182 |  0.03545 |
| Llama Guard 3 1B (AuditKIT) | base -> repaired (SafeTune)    | (3, 0)       | 0.25    |  0.5     |
| Llama Guard 3 8B (HF Job 2) | base->repaired                 | (0, 3)       | 0.25    |  0.5     |

