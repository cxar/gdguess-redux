# Architectural Decisions (append-only)

- M0: Enforce docs-drift guard to couple code and docs.
- M0: Assume intermingled audio sources; no label QA.
- M0: Accuracy-first; OpenBEATs-Large backbone planned.
- M2: MixStyle for domain adaptation - mixes feature statistics to reduce domain gap between recording conditions without needing domain labels.
- M2: Device coloration simulates consumer devices (phones, laptops) rather than studio equipment to match real-world deployment.
- M2: Synthetic noise generation over recorded noise banks - ensures determinism and avoids copyright/storage issues.
- M2: FIR approximation for device EQ - simpler than full IIR state management in batched training.
