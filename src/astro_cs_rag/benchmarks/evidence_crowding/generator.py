"""Generate synthetic bridge-hop datasets with controlled crowding.

Each query is a 2-hop bridge: ``film → director → country``. The gold
support chain is two atoms (hop1 + hop2). Distractors are drawn from
typed pools so we can vary entity overlap, claim-type overlap, and
semantic similarity independently of dense retrieval performance.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .schema import (
    CrowdingAtom,
    CrowdingCell,
    CrowdingChunk,
    CrowdingDataset,
    CrowdingQuery,
)

_FILMS = [
    "Liora's Map", "The Northern Light", "Salt Harbor", "Vellum Sky",
    "Ember Path", "The Iron Garden", "Quiet Crossing", "Glass Almanac",
    "Halcyon Run", "The Brass Compass",
]
_DIRECTORS = [
    "Mara Keene", "Arvid Lund", "Inga Solberg", "Paolo Renzi",
    "Hana Okabe", "Diego Marín", "Petra Dvorak", "Yusuf Demir",
    "Naoko Inoue", "Saffi Nilsson",
]
_COUNTRIES = [
    "Norway", "Italy", "Japan", "Portugal", "Hungary",
    "Greece", "Sweden", "Spain", "Czechia", "Turkey",
]
_PARAPHRASES_BORN = [
    "{d} was born in {c}.",
    "The director {d} hails from {c}.",
    "{d} is a native of {c}.",
    "{d}, originally from {c}, became a filmmaker.",
]
_PARAPHRASES_DIRECTED = [
    "{f} was directed by {d}.",
    "The film {f} is the work of director {d}.",
    "{d} directed {f}.",
    "{d} helmed the production of {f}.",
]


@dataclass(frozen=True)
class GoldTriple:
    film: str
    director: str
    country: str


def _make_gold_triples(n: int, rng: random.Random) -> list[GoldTriple]:
    out = []
    for i in range(n):
        out.append(
            GoldTriple(
                film=_FILMS[i % len(_FILMS)] + (f" {i//len(_FILMS)}" if i >= len(_FILMS) else ""),
                director=_DIRECTORS[i % len(_DIRECTORS)] + (f" {i}" if i >= len(_DIRECTORS) else ""),
                country=_COUNTRIES[i % len(_COUNTRIES)],
            )
        )
    rng.shuffle(out)
    return out


def _distractor_atoms(
    triple: GoldTriple,
    cell: CrowdingCell,
    base_id: str,
    rng: random.Random,
) -> list[CrowdingAtom]:
    n = cell.n_distractors_per_gold
    atoms: list[CrowdingAtom] = []
    for k in range(n):
        kind = ("entity_overlap", "type_overlap", "semantic", "noise")[k % 4]
        if kind == "entity_overlap":
            wrong_country = rng.choice([c for c in _COUNTRIES if c != triple.country])
            text = rng.choice(_PARAPHRASES_BORN).format(d=triple.director, c=wrong_country)
            ents = [triple.director, wrong_country]
            ctype = "WHERE"
        elif kind == "type_overlap":
            person = rng.choice([d for d in _DIRECTORS if d != triple.director])
            country = rng.choice(_COUNTRIES)
            text = rng.choice(_PARAPHRASES_BORN).format(d=person, c=country)
            ents = [person, country]
            ctype = "WHERE"
        elif kind == "semantic":
            wrong_country = rng.choice([c for c in _COUNTRIES if c != triple.country])
            text = f"The director of {triple.film} grew up in {wrong_country}."
            ents = [triple.film, wrong_country]
            ctype = "WHERE"
        else:  # noise
            text = f"{rng.choice(_FILMS)} premiered at a film festival in {rng.choice(_COUNTRIES)}."
            ents = []
            ctype = "ANY"
        atoms.append(
            CrowdingAtom(
                atom_id=f"{base_id}::dx{k}",
                chunk_id="",  # filled when packed into chunks
                doc_id="",
                text=text,
                claim_type=ctype,
                is_gold=False,
                role="distractor",
                distractor_class=kind,  # type: ignore[arg-type]
                entities=ents,
            )
        )
    return atoms


def _pack_into_chunks(
    atoms: list[CrowdingAtom], cell: CrowdingCell
) -> list[CrowdingChunk]:
    """Group atoms into carrier chunks based on `chunk_mixing` mode."""
    chunks: list[CrowdingChunk] = []
    if cell.chunk_mixing == "gold_isolated":
        for i, a in enumerate(atoms):
            cid = f"chunk_{i}"
            a.chunk_id = cid
            a.doc_id = f"doc_{i}" if a.is_gold else f"doc_distractor_{i}"
            chunks.append(
                CrowdingChunk(chunk_id=cid, doc_id=a.doc_id, text=a.text, atom_ids=[a.atom_id])
            )
        return chunks
    # gold_with_distractors / bridge_buried — pack ~chunk_size chars per chunk.
    cur: list[CrowdingAtom] = []
    cur_len = 0
    cidx = 0
    for a in atoms:
        if cur and cur_len + len(a.text) + 1 > cell.chunk_size:
            cid = f"chunk_{cidx}"
            for cc in cur:
                cc.chunk_id = cid
                cc.doc_id = f"doc_{cidx}"
            chunks.append(
                CrowdingChunk(
                    chunk_id=cid,
                    doc_id=f"doc_{cidx}",
                    text=" ".join(c.text for c in cur),
                    atom_ids=[c.atom_id for c in cur],
                )
            )
            cur, cur_len, cidx = [], 0, cidx + 1
        cur.append(a)
        cur_len += len(a.text) + 1
    if cur:
        cid = f"chunk_{cidx}"
        for cc in cur:
            cc.chunk_id = cid
            cc.doc_id = f"doc_{cidx}"
        chunks.append(
            CrowdingChunk(
                chunk_id=cid,
                doc_id=f"doc_{cidx}",
                text=" ".join(c.text for c in cur),
                atom_ids=[c.atom_id for c in cur],
            )
        )
    return chunks


def build_dataset(
    cell: CrowdingCell, n_queries: int, rng: random.Random | None = None
) -> CrowdingDataset:
    """Build one cell of the sweep — `n_queries` synthetic bridge questions."""
    rng = rng or random.Random(cell.seed)
    triples = _make_gold_triples(n_queries, rng)
    all_atoms: list[CrowdingAtom] = []
    queries: list[CrowdingQuery] = []
    for qi, t in enumerate(triples):
        qid = f"{cell.cell_id}::q{qi}"
        hop1_text = rng.choice(_PARAPHRASES_DIRECTED).format(f=t.film, d=t.director)
        hop2_text = rng.choice(_PARAPHRASES_BORN).format(d=t.director, c=t.country)
        hop1 = CrowdingAtom(
            atom_id=f"{qid}::hop1", chunk_id="", doc_id="",
            text=hop1_text, claim_type="WHO", is_gold=True, role="hop1",
            entities=[t.film, t.director],
        )
        hop2 = CrowdingAtom(
            atom_id=f"{qid}::hop2", chunk_id="", doc_id="",
            text=hop2_text, claim_type="WHERE", is_gold=True, role="hop2",
            entities=[t.director, t.country],
        )
        gold_atoms = [hop1, hop2]
        distractors = _distractor_atoms(t, cell, qid, rng)
        # bridge_buried: place distractors between hop1 and hop2.
        if cell.chunk_mixing == "bridge_buried" and distractors:
            ordered = [hop1, *distractors, hop2]
        else:
            ordered = [hop1, hop2, *distractors]
        all_atoms.extend(ordered)
        queries.append(
            CrowdingQuery(
                query_id=qid,
                text=f"What country was the director of {t.film} born in?",
                answer=t.country,
                answer_aliases=[t.country, t.country.lower()],
                gold_atom_ids=[hop1.atom_id, hop2.atom_id],
                gold_doc_ids=[],  # filled after chunking
                hop_count=2,
                template="film_director_country",
            )
        )
    chunks = _pack_into_chunks(all_atoms, cell)
    # populate gold_doc_ids per query from gold atoms after chunking.
    by_atom = {a.atom_id: a for a in all_atoms}
    for q in queries:
        q.gold_doc_ids = sorted({by_atom[aid].doc_id for aid in q.gold_atom_ids})
    return CrowdingDataset(cell=cell, atoms=all_atoms, chunks=chunks, queries=queries)
