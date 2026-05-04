# Astro Detector Module

Inputs:
- candidate records
- chunk records
- query record

Steps:
1. compute or read raw similarity
2. estimate local background
3. compute SNR score
4. apply threshold or rank
5. emit detected evidence candidates

Outputs must include:
- raw score
- background mean
- background std
- evidence-SNR
- detector rank
