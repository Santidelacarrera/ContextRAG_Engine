# Security policy

## Reporting a vulnerability

Do not open a public issue. Report it privately through GitHub's *Security → Report a vulnerability* on this repository,
with the affected version or commit, a reproduction, and the impact you observed.

You will get an acknowledgement and an assessment; fixes are released as a patch version and listed in `CHANGELOG.md`.

## Scope and known limits

The threat model, the encryption-at-rest coverage, the known gaps (prompt-injection detection, vector/metadata
encryption, TLS termination left to the deployment) and the pre-pilot checklist are in [docs/PILOT.md](docs/PILOT.md).
Please read them before reporting something that is documented there as a known limit.
