# Config Contracts

Config files should define:
- dataset
- corpus path
- query path
- output path
- seed
- retriever type
- embedding model
- candidate top-N
- detector settings
- selector settings
- generator settings
- metric settings

Use YAML for human editing.
Validate with Pydantic before running.
