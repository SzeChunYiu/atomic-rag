# Chunker Module

Responsibilities:
- split documents into chunks
- preserve source offsets
- preserve document metadata
- estimate token count
- write chunk manifest

Chunk fields:
- chunk_id
- doc_id
- text
- start_char
- end_char
- token_count
- metadata

Sanity checks:
- no empty chunks
- no lost text unless configured
- offsets map back to source
