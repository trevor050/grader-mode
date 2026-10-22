# Pulse Round 1 status

- Before rental: credit $3.845357334199992; no live Vast instances. Existing unrelated edits: explore/attrib2.py, explore/rlprompt3/, notes/jpp-lens.md; preserved.
- Protocol: explore/PULSE.md, to be committed and pushed before any rental.
- 2026-10-22 05:44:16 UTC: rented A100 SXM4 80GB instance 57166538, 150GB disk; quoted total $1.1083333333/hour, ingress $0.0053333333/GB, egress $0.008/GB, reliability 0.992706. Protocol already pushed at 7734fe6. Persistent local guard uses $1.20/hour plus $0.45 network reserve; destroys below $0.40 or >900s without progress. No top-ups configured.
- 05:49 UTC: container running; setup and model download launched via nohup. CLI image boot took approximately 5 minutes. Scripts preserve all 864 planned capture conditions and exact generation token IDs.
