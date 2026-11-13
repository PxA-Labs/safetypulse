## A. AUROC at the pre-registered cut-off (base + 10 pts)

| set                        | signal       |   auroc | run_boot_95    |   valid_resamples |   checkpoints |   unsafe |   runs |
|:---------------------------|:-------------|--------:|:---------------|------------------:|--------------:|---------:|-------:|
| test, all checkpoints      | cis_circ     |   0.931 | (0.873, 1.0)   |              2000 |            77 |       23 |      7 |
| test, all checkpoints      | cis_sep      |   0.957 | (0.947, 1.0)   |              2000 |            77 |       23 |      7 |
| test, all checkpoints      | cis_cos      |   0.959 | (0.898, 1.0)   |              2000 |            77 |       23 |      7 |
| test, all checkpoints      | cis_logit    |   0.634 | (0.593, 0.684) |              2000 |            77 |       23 |      7 |
| test, all checkpoints      | spot_refusal |   0.961 | (0.915, 1.0)   |              2000 |            77 |       23 |      7 |
| test, all checkpoints      | dw_norm      |   0.737 | (0.696, 0.836) |              2000 |            77 |       23 |      7 |
| test, all checkpoints      | step         |   0.686 | (0.64, 0.757)  |              2000 |            77 |       23 |      7 |
| test, step > 0             | cis_circ     |   0.92  | (0.856, 1.0)   |              2000 |            70 |       23 |      7 |
| test, step > 0             | cis_sep      |   0.951 | (0.939, 1.0)   |              2000 |            70 |       23 |      7 |
| test, step > 0             | cis_cos      |   0.953 | (0.885, 1.0)   |              2000 |            70 |       23 |      7 |
| test, step > 0             | cis_logit    |   0.593 | (0.538, 0.636) |              2000 |            70 |       23 |      7 |
| test, step > 0             | spot_refusal |   0.956 | (0.904, 1.0)   |              2000 |            70 |       23 |      7 |
| test, step > 0             | dw_norm      |   0.698 | (0.648, 0.814) |              2000 |            70 |       23 |      7 |
| test, step > 0             | step         |   0.639 | (0.581, 0.724) |              2000 |            70 |       23 |      7 |
| test Alpaca runs, step > 0 | cis_circ     |   0.466 | (0.375, 1.0)   |              2000 |            30 |       23 |      3 |
| test Alpaca runs, step > 0 | cis_sep      |   0.671 | (0.625, 1.0)   |              2000 |            30 |       23 |      3 |
| test Alpaca runs, step > 0 | cis_cos      |   0.683 | (0.5, 1.0)     |              2000 |            30 |       23 |      3 |
| test Alpaca runs, step > 0 | cis_logit    |   0.497 | (0.25, 1.0)    |              2000 |            30 |       23 |      3 |
| test Alpaca runs, step > 0 | spot_refusal |   0.702 | (0.583, 1.0)   |              2000 |            30 |       23 |      3 |
| test Alpaca runs, step > 0 | dw_norm      |   0.888 | (0.873, 1.0)   |              2000 |            30 |       23 |      3 |
| test Alpaca runs, step > 0 | step         |   0.901 | (0.889, 1.0)   |              2000 |            30 |       23 |      3 |

## B. Paired AUROC differences (base + 10 pts)

| set                        | difference              |   delta_auroc | run_boot_95    |   valid_resamples |
|:---------------------------|:------------------------|--------------:|:---------------|------------------:|
| test, all checkpoints      | cis_sep − cis_circ      |         0.027 | (0.0, 0.093)   |              2000 |
| test, all checkpoints      | cis_circ − dw_norm      |         0.194 | (0.037, 0.296) |              2000 |
| test, all checkpoints      | cis_sep − dw_norm       |         0.221 | (0.13, 0.296)  |              2000 |
| test, all checkpoints      | cis_circ − spot_refusal |        -0.031 | (-0.042, 0.0)  |              2000 |
| test, all checkpoints      | cis_circ − step         |         0.245 | (0.116, 0.36)  |              2000 |
| test, step > 0             | cis_sep − cis_circ      |         0.031 | (0.0, 0.106)   |              2000 |
| test, step > 0             | cis_circ − dw_norm      |         0.223 | (0.042, 0.344) |              2000 |
| test, step > 0             | cis_sep − dw_norm       |         0.253 | (0.147, 0.344) |              2000 |
| test, step > 0             | cis_circ − spot_refusal |        -0.035 | (-0.048, 0.0)  |              2000 |
| test, step > 0             | cis_circ − step         |         0.281 | (0.131, 0.419) |              2000 |
| test Alpaca runs, step > 0 | cis_sep − cis_circ      |         0.205 | (0.0, 0.458)   |              2000 |
| test Alpaca runs, step > 0 | cis_circ − dw_norm      |        -0.422 | (-0.542, 0.0)  |              2000 |
| test Alpaca runs, step > 0 | cis_sep − dw_norm       |        -0.217 | (-0.312, 0.0)  |              2000 |
| test Alpaca runs, step > 0 | cis_circ − spot_refusal |        -0.236 | (-0.281, 0.0)  |              2000 |
| test Alpaca runs, step > 0 | cis_circ − step         |        -0.435 | (-0.542, 0.0)  |              2000 |

## C. Sensitivity to the unsafe cut-off

|                                                |     5 |    10 |    15 |
|:-----------------------------------------------|------:|------:|------:|
| ('test Alpaca runs, step > 0', 'cis_circ')     | 0.663 | 0.466 | 0.545 |
| ('test Alpaca runs, step > 0', 'cis_cos')      | 0.74  | 0.683 | 0.762 |
| ('test Alpaca runs, step > 0', 'cis_logit')    | 0.683 | 0.497 | 0.529 |
| ('test Alpaca runs, step > 0', 'cis_sep')      | 0.894 | 0.671 | 0.587 |
| ('test Alpaca runs, step > 0', 'dw_norm')      | 0.981 | 0.888 | 0.582 |
| ('test Alpaca runs, step > 0', 'spot_refusal') | 0.971 | 0.702 | 0.698 |
| ('test Alpaca runs, step > 0', 'step')         | 0.99  | 0.901 | 0.571 |
| ('test, all checkpoints', 'cis_circ')          | 0.974 | 0.931 | 0.859 |
| ('test, all checkpoints', 'cis_cos')           | 0.98  | 0.959 | 0.926 |
| ('test, all checkpoints', 'cis_logit')         | 0.669 | 0.634 | 0.634 |
| ('test, all checkpoints', 'cis_sep')           | 0.992 | 0.957 | 0.873 |
| ('test, all checkpoints', 'dw_norm')           | 0.711 | 0.737 | 0.641 |
| ('test, all checkpoints', 'spot_refusal')      | 0.998 | 0.961 | 0.907 |
| ('test, all checkpoints', 'step')              | 0.658 | 0.686 | 0.603 |
| ('test, step > 0', 'cis_circ')                 | 0.969 | 0.92  | 0.843 |
| ('test, step > 0', 'cis_cos')                  | 0.976 | 0.953 | 0.918 |
| ('test, step > 0', 'cis_logit')                | 0.628 | 0.593 | 0.617 |
| ('test, step > 0', 'cis_sep')                  | 0.99  | 0.951 | 0.858 |
| ('test, step > 0', 'dw_norm')                  | 0.665 | 0.698 | 0.599 |
| ('test, step > 0', 'spot_refusal')             | 0.997 | 0.956 | 0.896 |
| ('test, step > 0', 'step')                     | 0.604 | 0.639 | 0.557 |
