# Benchmark Snapshot

Current head-to-head results collected during the Owen vs. MC-MoE comparison:

| Task | Owen | MC-MoE | Winner |
| --- | ---: | ---: | --- |
| BLINK | 0.4139926354550237 | 0.398211467648606 | Owen |
| RealWorldQA | 0.6457516339869281 | 0.665359477124183 | MC-MoE |
| ScienceQA_VAL | 0.8516928946113496 | 0.8655221745350501 | MC-MoE |
| ScienceQA_TEST | 0.8701041150223103 | 0.8750619732275657 | MC-MoE |
| DocVQA_VAL | 87.403 | 88.186 | MC-MoE |
| HallusionBench | 55.3102 | 57.3081 | MC-MoE |
| POPE | 87.09025594970812 | 87.77422555396318 | MC-MoE |
| CMMMU_VAL | 0.276667 | 0.285556 | MC-MoE |
| A-OKVQA | 0.794253938832252 | 0.7956331877729258 | MC-MoE |
| VizWiz | 6.58486 | 6.29312 | Owen |
| GQA_TestDev_Balanced | 61.3055 | 61.7268 | MC-MoE |

MME breakdown:

| Metric | Owen | MC-MoE | Winner |
| --- | ---: | ---: | --- |
| MME perception | 1565.59964 | 1539.201881 | Owen |
| MME reasoning | 345.357143 | 353.214286 | MC-MoE |
| MME total | 1910.956783 | 1892.416167 | Owen |

Additional note:

- `MMMU_DEV_VAL` was previously observed as an Owen win in the broader experiment history, but it is not re-run inside this trimmed repo snapshot.
- `InfoVQA_TEST` supports inference only in the local setup because the test split has no public ground truth.
- `DocVQA_TEST` failed locally during dataset construction in this run and is not included as a scored comparison.
