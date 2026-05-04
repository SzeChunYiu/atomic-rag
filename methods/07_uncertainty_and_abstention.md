# Uncertainty and Abstention

The system should know when evidence is weak.

Uncertainty signals:
- low max SNR
- high disagreement among candidates
- missing query facets
- selected evidence has low support
- contradiction penalty is high
- generator creates unsupported claims

Possible actions:
- retrieve more candidates
- switch to multi-hop mode
- answer with caveat
- abstain
- ask for clarification in interactive setting
