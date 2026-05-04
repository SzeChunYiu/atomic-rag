"""IRC-robustness, iteration 2 — larger synthetic experiment with bootstrap power.

Scales iter 1 along three axes to lift directional signal (P=0.76) above p<0.01:
- queries: 30 → ~120 (templated generation)
- chunk_sizes: 6 → 20 (50..300 step ~13)
- seeds: 1 → 5 (varies distractor sampling)

Records per-(seed, chunk_size, selector) gold_pair_coverage and runs a paired
bootstrap natively. The statistic we care about is *chunk-size stdev within a
seed*: anti_kt should have a tighter distribution than greedy across chunk
sizes, robustly across seeds.

Stays under the 300-line cap by delegating corpus/config plumbing to iter 1's
helpers.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from astro_cs_rag.config.loader import load_yaml  # noqa: E402
from astro_cs_rag.config.schema import BenchmarkConfig  # noqa: E402
from astro_cs_rag.pipeline.benchmark import benchmark_run  # noqa: E402
from synthetic_irc_experiment import (  # noqa: E402
    _TOPICS,
    base_cfg,
    gold_pair_coverage,
)

REPO = Path(__file__).resolve().parents[1]


_EXTRA_TOPICS = [
    ("axion-like particle", "haloscope cavity resonance", "narrow microwave excess"),
    ("cosmic dawn", "Lyman-alpha forest", "first generation stars"),
    ("supercluster filament", "warm hot intergalactic medium", "missing baryons"),
    ("tidal disruption event", "main sequence star encounter", "X-ray flare lightcurve"),
    ("magnetorotational instability", "Maxwell stress", "angular momentum transport"),
    ("BL Lac object", "featureless continuum", "synchrotron self compton"),
    ("Wolf-Rayet star", "fast stellar wind", "carbon-rich atmosphere"),
    ("X-ray polarimetry", "vacuum birefringence", "magnetar surface field"),
    ("strong gravitational lensing", "Einstein ring", "time delay cosmography"),
    ("ultradiffuse galaxy", "low surface brightness", "globular cluster richness"),
    ("Sagittarius A*", "event horizon shadow", "millimeter VLBI image"),
    ("Lyman-alpha emitter", "narrowband imaging", "high redshift survey"),
    ("massive star binary", "Wolf-Rayet progenitor", "GW progenitor channel"),
    ("dark matter direct detection", "xenon time projection chamber", "WIMP recoil signal"),
    ("solar dynamo", "differential rotation shear", "11 year cycle"),
    ("interstellar molecular cloud", "CO emission line", "star formation tracer"),
    ("planetary nebula", "asymptotic giant branch progenitor", "envelope ejection"),
    ("gravitational wave inspiral", "compact binary coalescence", "chirp mass measurement"),
    ("quasar absorption system", "damped Lyman alpha", "neutral hydrogen column"),
    ("cosmic shear survey", "weak lensing tomography", "matter clustering amplitude"),
    ("eclipsing binary", "primary minimum depth", "mass ratio constraint"),
    ("Schwarzschild orbit", "innermost stable circular orbit", "spin measurement"),
    ("intracluster medium", "X-ray surface brightness", "metallicity profile"),
    ("brown dwarf atmosphere", "methane absorption band", "T dwarf classification"),
    ("solar wind", "interplanetary magnetic field", "Alfven wave heating"),
    ("primordial nucleosynthesis", "deuterium abundance", "baryon density constraint"),
    ("dark matter substructure", "stellar stream perturbation", "subhalo mass function"),
    ("pulsar wind nebula", "termination shock", "synchrotron cooling break"),
    ("stellar oscillation", "asteroseismic mode pattern", "convective core size"),
    ("microlensing event", "Einstein ring crossing time", "lens mass distribution"),
    ("relativistic jet", "Lorentz factor estimate", "synchrotron self absorption"),
    ("neutrino oscillation", "mass eigenstate splitting", "long baseline experiment"),
    ("interstellar shock", "Sedov-Taylor blast wave", "thermal X-ray emission"),
    ("galactic center", "stellar S-cluster orbits", "supermassive black hole mass"),
    ("Faraday rotation", "magnetized plasma traversal", "polarization angle rotation"),
    ("dust grain alignment", "radiative torque mechanism", "polarized far-infrared emission"),
    ("supernova remnant cooling", "post-shock plasma", "non-equilibrium ionization"),
    ("planet formation disk", "protoplanetary gap", "ALMA dust continuum"),
    ("globular cluster blue stragglers", "binary mass transfer", "stellar rejuvenation"),
    ("ultra-faint dwarf galaxy", "metal poor star", "early universe enrichment"),
    ("solar coronal heating", "nanoflare reconnection", "EUV emission excess"),
    ("X-ray pulsar accretion column", "Compton scattering hot spot", "cyclotron resonance feature"),
    ("Type II supernova plateau", "hydrogen envelope recombination", "shock cooling tail"),
    ("Hubble tension", "local distance ladder", "early universe inference disagreement"),
    ("primordial gravitational wave", "B-mode polarization", "tensor amplitude bound"),
    ("microquasar accretion state", "low hard transition", "radio jet ejection"),
    ("hot Jupiter atmosphere", "transmission spectroscopy", "sodium absorption signature"),
    ("solar prominence", "filament eruption", "coronal mass ejection trigger"),
    ("interstellar magnetic field", "Zeeman splitting", "molecular cloud field strength"),
    ("Cherenkov telescope array", "TeV gamma ray detection", "imaging atmospheric shower"),
    ("gravitational microlensing planet", "caustic crossing anomaly", "planet host mass"),
    ("stellar nucleosynthesis r-process", "neutron capture rapid", "heavy element synthesis"),
    ("dark matter annihilation", "gamma ray excess", "Galactic center signal"),
    ("compact object kick", "natal asymmetric explosion", "system orbit eccentricity"),
    ("interstellar plasma", "dispersion measure", "fast radio burst distance"),
    ("solar neutrino flux", "boron 8 reaction chain", "Super-Kamiokande measurement"),
    ("galactic warm ionized medium", "H-alpha diffuse emission", "scale height pressure"),
    ("AGN feedback", "radio mode jet inflation", "intracluster cavity formation"),
    ("blazar polarization swing", "geometric helical jet", "rotation event"),
    ("binary neutron star merger", "GW170817 multimessenger", "kilonova lanthanide opacity"),
    ("strong-field QED", "Schwinger pair production", "magnetar surface vacuum"),
    ("solar coronal mass ejection", "magnetic flux rope", "interplanetary disturbance"),
    ("dark energy survey", "Type Ia supernova distance", "equation of state constraint"),
    ("galactic chemical evolution", "alpha-element ratio", "supernova yield tracer"),
    ("stellar magnetic activity", "chromospheric Ca II line", "rotation rate proxy"),
    ("gamma-ray pulsar", "outer gap acceleration", "Fermi LAT detection"),
    ("dark matter sub-GeV scattering", "phonon excitation detector", "low threshold limit"),
    ("merger tree halo", "extended Press-Schechter", "progenitor mass function"),
    ("solar flare hard X-ray", "thick target bremsstrahlung", "non-thermal electron beam"),
    ("interstellar bubble", "stellar wind feedback", "shell expansion velocity"),
    ("relativistic shock breakout", "supernova first light", "UV flash signature"),
    ("Type Ic broad-line supernova", "GRB associated event", "high explosion energy"),
    ("solar wind termination", "heliopause crossing", "Voyager interstellar plasma"),
    ("intergalactic magnetic field", "blazar pair halo", "lower bound estimate"),
    ("binary black hole inspiral", "tidal heating", "post-Newtonian phase"),
    ("dark matter direct detection xenon", "S2-only ionization", "low energy threshold"),
    ("radio-loud AGN unification", "viewing angle dependence", "type I type II distinction"),
    ("circumstellar disk turbulence", "magnetorotational instability", "alpha viscosity"),
    ("ultra-high energy cosmic ray anisotropy", "Pierre Auger dipole", "extragalactic origin"),
    ("solar oscillation p-mode", "acoustic standing wave", "internal rotation diagnostic"),
    ("stellar collapse core bounce", "neutrino burst signal", "supernova trigger"),
    ("AGN reverberation mapping", "broad line region size", "black hole mass scaling"),
    ("solar polar magnetic field", "11 year reversal", "next cycle prediction"),
    ("extreme mass ratio inspiral", "LISA detection", "Kerr spacetime test"),
    ("cosmic ray air shower", "muon excess problem", "hadronic model tension"),
    ("dark photon kinetic mixing", "fixed target experiment", "displaced vertex search"),
    ("interstellar PAH emission", "UV pumped fluorescence", "mid-infrared band"),
    ("magnetar giant flare", "fireball ejection", "afterglow gamma-ray tail"),
    ("Type Ia supernova progenitor", "double degenerate channel", "delay time distribution"),
    ("X-ray reverberation lag", "inner disk reflection", "black hole spin probe"),
    ("solar quiet network", "internetwork magnetic flux", "small scale dynamo"),
    ("stellar binary common envelope", "spiral-in inspiral", "orbital energy transfer"),
    ("dark matter halo concentration", "mass-concentration relation", "redshift evolution"),
]


def _all_topics() -> list[tuple[str, str, str]]:
    return list(_TOPICS) + _EXTRA_TOPICS


def build_corpus(out_dir: Path, *, n_topics: int, n_distractors: int, seed: int) -> dict[str, Path]:
    """Templated synthetic corpus, parametric in n_topics and seed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    topics = _all_topics()[:n_topics]
    rng = np.random.default_rng(seed)
    docs: list[dict] = []
    queries: list[dict] = []
    golds: list[dict] = []

    for i, (topic, sent_a, sent_b) in enumerate(topics):
        doc_id = f"gold_{i:04d}"
        text = (
            f"{topic.title()} is associated with {sent_a}. "
            f"In particular, observations of {topic} reveal {sent_b}."
        )
        docs.append({"doc_id": doc_id, "text": text, "metadata": {"role": "gold", "answer_phrase": sent_b}})
        qid = f"q_{i:04d}"
        queries.append({
            "query_id": qid,
            "text": f"What does {topic} reveal about its physical mechanism?",
            "gold_doc_ids": [doc_id],
            "metadata": {"answer": [sent_b], "topic": topic},
        })
        golds.append({"query_id": qid, "gold_doc_ids": [doc_id]})

    distractor_topics = [
        "linear algebra factorization", "convex optimization duality", "graph coloring",
        "sorting networks", "register allocation heuristics", "fluid mixing dynamics",
        "polymer chain statistics", "epidemic threshold models", "queueing theory bounds",
        "auction mechanism design", "lattice gas automata", "membrane permeability scaling",
        "tensor decomposition", "spectral graph theory", "compressed sensing recovery",
        "randomized numerical linear algebra", "cache-oblivious algorithms",
    ]
    rng.shuffle(distractor_topics)
    for j in range(n_distractors):
        t = distractor_topics[j % len(distractor_topics)]
        docs.append({
            "doc_id": f"dist_{j:05d}",
            "text": (
                f"This passage on {t} explores theoretical ideas. "
                f"Researchers in {t} examine convergence rates and asymptotic limits."
            ),
            "metadata": {"role": "distractor"},
        })

    (out_dir / "corpus.jsonl").write_text("\n".join(json.dumps(d) for d in docs) + "\n", encoding="utf-8")
    (out_dir / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in queries) + "\n", encoding="utf-8")
    (out_dir / "gold.jsonl").write_text("\n".join(json.dumps(g) for g in golds) + "\n", encoding="utf-8")
    return {"corpus": out_dir / "corpus.jsonl", "queries": out_dir / "queries.jsonl", "gold": out_dir / "gold.jsonl"}


def run_one(out_root: Path, paths: dict[str, Path], cs: int, sel: str, seed: int) -> dict:
    run_out = out_root / f"seed{seed}_cs{cs}_{sel}"
    run_out.mkdir(parents=True, exist_ok=True)
    cfg_dict = base_cfg(run_out, paths, chunk_size=cs, mode=sel)
    cfg_dict["seed"] = int(seed)
    tmp = run_out / "tmp"
    tmp.mkdir(exist_ok=True)
    cfg_path = tmp / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")
    cfg = load_yaml(cfg_path, BenchmarkConfig)
    run_dir = benchmark_run(cfg)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    cov = gold_pair_coverage(run_dir, paths["gold"], paths["queries"])
    return {
        "seed": seed,
        "chunk_size": cs,
        "selector": sel,
        "recall@1": float(metrics.get("recall@1_doc_mean", 0.0)),
        "answer_f1": float(metrics.get("answer_f1_mean", 0.0)),
        "gold_pair_coverage": cov,
    }


def paired_bootstrap(rows: list[dict], a: str, b: str, *, n_resamples: int = 5000, seed: int = 0) -> dict:
    """Stratified paired bootstrap: resample chunk_sizes within each seed,
    compute stdev-of-coverage across chunk_sizes for selectors a and b, and
    measure P(stdev_a < stdev_b) and the mean paired diff.
    """
    seeds = sorted({r["seed"] for r in rows})
    chunk_sizes = sorted({r["chunk_size"] for r in rows})
    by_key: dict[tuple[int, int, str], float] = {(r["seed"], r["chunk_size"], r["selector"]): r["gold_pair_coverage"] for r in rows}
    rng = np.random.default_rng(seed)
    diffs = []
    a_lower = 0
    for _ in range(n_resamples):
        idx = rng.integers(0, len(chunk_sizes), size=len(chunk_sizes))
        sampled_cs = [chunk_sizes[i] for i in idx]
        std_a_per_seed = []
        std_b_per_seed = []
        for s in seeds:
            va = [by_key[(s, cs, a)] for cs in sampled_cs if (s, cs, a) in by_key]
            vb = [by_key[(s, cs, b)] for cs in sampled_cs if (s, cs, b) in by_key]
            if len(va) >= 2 and len(vb) >= 2:
                std_a_per_seed.append(statistics.pstdev(va))
                std_b_per_seed.append(statistics.pstdev(vb))
        if not std_a_per_seed:
            continue
        sa = float(np.mean(std_a_per_seed))
        sb = float(np.mean(std_b_per_seed))
        diffs.append(sa - sb)
        if sa < sb:
            a_lower += 1
    diffs = np.asarray(diffs)
    return {
        "selector_a": a,
        "selector_b": b,
        "p_a_lower_stdev": float(a_lower / max(1, len(diffs))),
        "diff_mean": float(diffs.mean()),
        "diff_p05": float(np.percentile(diffs, 5)),
        "diff_p95": float(np.percentile(diffs, 95)),
        "n_resamples": int(len(diffs)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "runs/synthetic_irc_iter2")
    ap.add_argument("--n-topics", type=int, default=120)
    ap.add_argument("--n-distractors", type=int, default=240)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--chunk-min", type=int, default=50)
    ap.add_argument("--chunk-max", type=int, default=300)
    ap.add_argument("--chunk-steps", type=int, default=20)
    ap.add_argument("--selectors", nargs="+", default=["greedy", "anti_kt"])
    args = ap.parse_args()

    chunk_sizes = [int(x) for x in np.linspace(args.chunk_min, args.chunk_max, args.chunk_steps).round().tolist()]
    chunk_sizes = sorted(set(chunk_sizes))
    rows: list[dict] = []
    for seed in range(args.n_seeds):
        corpus_dir = REPO / f"data/synthetic_irc_iter2/seed_{seed}"
        paths = build_corpus(corpus_dir, n_topics=args.n_topics, n_distractors=args.n_distractors, seed=seed)
        for cs in chunk_sizes:
            for sel in args.selectors:
                row = run_one(args.out, paths, cs, sel, seed)
                rows.append(row)
                print(f"seed={seed} cs={cs} sel={sel} cov={row['gold_pair_coverage']:.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    import csv
    with (args.out / "results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    boot = {}
    if "anti_kt" in args.selectors and "greedy" in args.selectors:
        boot["anti_vs_greedy"] = paired_bootstrap(rows, "anti_kt", "greedy")
    if "anti_kt" in args.selectors and "mmr" in args.selectors:
        boot["anti_vs_mmr"] = paired_bootstrap(rows, "anti_kt", "mmr")
    (args.out / "bootstrap.json").write_text(json.dumps(boot, indent=2), encoding="utf-8")
    print(json.dumps(boot, indent=2))


if __name__ == "__main__":
    main()
