# Atomic RAG Theory

RAG should be decomposed into atoms before new methods are designed.

Atomic objects:
- query text
- latent information need
- document
- chunk
- span
- evidence atom
- background noise
- candidate set
- selected evidence set
- generated claim
- citation
- metric

Evidence atom:
A minimal unit of support for an answer claim.

Evidence atom fields:
- claim
- source id
- span location
- relation to query
- support strength
- uncertainty
- reliability
- contradiction links

Atomic rule:
A chunk is only useful if it contains or points to evidence atoms.
