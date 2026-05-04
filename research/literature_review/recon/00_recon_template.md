# Recon template — proposed method

Filename convention: `NN_<short-method-name>.md`. One file per method. Kept
short (<80 lines). The point is rapid triage, not a survey.

## 1. Method one-liner
What we are proposing in 1–2 sentences, in terms a reviewer can grep.

## 2. Physics analog (precise)
The exact physics method we are borrowing — name, canonical reference,
mathematical form. Distinguish *operator-level* identity from *metaphor*.

## 3. Closest prior art
List up to five most-related published works in IR/NLP/ML.
Each entry: (year) author venue, one-sentence summary, *closest specific
overlap* with our method, and the **distinct** thing we add.

## 4. Novelty estimate
| dimension | grade |
|---|---|
| algorithmic novelty | low / medium / high |
| theoretical novelty | low / medium / high |
| empirical novelty (benchmarks/protocols) | low / medium / high |

Grade with a one-line justification. Do not promote.

## 5. Why this is publishable to NMI / NeurIPS / SIGIR
A reviewer-friendly hook: what existing failure mode (F1–F9) we attack, what
theorem-shaped statement we can prove, what cross-domain validation we can
provide.

## 6. Falsification protocol
How we would *kill* this method if it doesn't work.
Concretely: experiment, threshold, decision rule.

## 7. Required dependencies and risks
External libraries, datasets, GPU minutes, calibration data.

## 8. Status
- [ ] prior-art search done
- [ ] minimal implementation sketch
- [ ] ablation plan written
- [ ] first benchmark numbers
- [ ] falsification run
