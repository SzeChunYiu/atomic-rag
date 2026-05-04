# Evidence-SNR Math

Base score:

```text
z_i = (s_i - mu_Bi) / (sigma_Bi + eps)
```

Where:
- s_i = relevance score for candidate i
- mu_Bi = local background mean
- sigma_Bi = local background std
- eps = small stability constant

Variants:
- length-normalized z
- reliability-weighted z
- query-facet z
- entity-match adjusted z

Required logs:
Store every component, not only final z.
