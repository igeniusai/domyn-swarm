# Security Policy

## Reporting a vulnerability

Report security vulnerabilities to **security@domyn.com**.

Please do not open a public GitHub issue, discussion or pull request for a
security report.

Include whatever you have:

- the version or commit affected
- the kind of issue — credential exposure, remote code execution, privilege
  escalation, denial of service, and so on
- steps to reproduce, and a proof of concept if you have one
- the impact, and how an attacker would reach it

We aim to acknowledge reports within five working days and will keep you
updated while we investigate. When a fix ships we will credit you in the
release notes unless you would rather we did not.

## Supported versions

domyn-swarm is pre-1.0 and under active development. Security fixes land on
`main` and in the next release; we do not backport them to earlier tags.

## Scope

domyn-swarm launches containers and submits jobs on infrastructure you control.
In scope: how domyn-swarm itself handles credentials and API tokens, what it
writes into generated job scripts and configuration, and how it submits work to
a backend.

Vulnerabilities in the components it drives — Slurm, Singularity/Apptainer,
vLLM, Ray, NVIDIA DGX Cloud Lepton — belong to those projects and should go to
their own security contacts.
