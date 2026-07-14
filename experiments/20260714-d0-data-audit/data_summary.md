# D0 Data Summary

## Official Dataset Medians

| dataset | images | brightness_median | contrast_median | blur_laplacian_median | aspect_long_short_median | megapixels_median | gt_regions_median |
| ------- | ------ | ----------------- | --------------- | --------------------- | ------------------------ | ----------------- | ----------------- |
| train   | 3272   | 0.5744            | 0.1838          | 1863.6321             | 1.3333                   | 1.2288            | 113.5000          |
| val     | 404    | 0.5709            | 0.1832          | 1764.1343             | 1.3333                   | 1.2288            | 110.5000          |
| test    | 413    | 0.5735            | 0.1885          | 1781.7480             | 1.3333                   | 1.2288            | 0.0000            |

## Largest Train-to-Target Distribution Shifts

| target             | metric            | reference_median | target_median | ks_statistic |
| ------------------ | ----------------- | ---------------- | ------------- | ------------ |
| pseudo_sroie       | brightness        | 0.5744           | 0.9384        | 0.9514       |
| pseudo_sroie       | saturation        | 0.1660           | 0.0000        | 0.9168       |
| pseudo_sroie       | entropy           | 6.9568           | 3.4291        | 0.9006       |
| pseudo_cord-v2     | edge_density      | 0.0812           | 0.0323        | 0.7636       |
| pseudo_cord-v2     | blur_laplacian    | 1863.6321        | 423.3710      | 0.7501       |
| pseudo_wildreceipt | megapixels        | 1.2288           | 0.2123        | 0.7070       |
| pseudo_cord-v2     | aspect_long_short | 1.3333           | 1.5000        | 0.5636       |
| pseudo_wildreceipt | aspect_long_short | 1.3333           | 1.3333        | 0.3458       |
| pseudo_wildreceipt | entropy           | 6.9568           | 6.6524        | 0.2455       |
| test               | edge_density      | 0.0812           | 0.0800        | 0.0655       |
| val                | blur_laplacian    | 1863.6321        | 1764.1343     | 0.0613       |
| test               | aspect_long_short | 1.3333           | 1.3333        | 0.0610       |
| test               | megapixels        | 1.2288           | 1.2288        | 0.0610       |
| val                | edge_density      | 0.0812           | 0.0783        | 0.0473       |
| val                | aspect_long_short | 1.3333           | 1.3333        | 0.0408       |

KS statistics rank distribution differences but do not by themselves prove that an
augmentation will improve CLEval. Adoption still requires controlled local evaluation.
