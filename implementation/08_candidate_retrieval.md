# Candidate Retrieval Module

Inputs:
- query record
- retriever config
- index

Outputs:
- Candidate records

Candidate fields:
- query_id
- chunk_id
- raw_score
- retriever_name
- rank
- latency_ms

Do not apply SNR here.
Do not select context here.
This module only creates the candidate pool.
