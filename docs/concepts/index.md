# Concepts

Explanation rather than instruction: why domyn-swarm is built the way it is. Read
these before changing the code, or when a guide's advice seems arbitrary.

- [Architecture](architecture.md) — the components and how they fit together
- [Serving vs compute backends](backends.md) — two protocols, and why not one
- [The SwarmJob lifecycle](swarmjob-lifecycle.md) — from CLI invocation to output file
- [Watchdog and collector](watchdog-collector.md) — why health reporting has its own process
- [Configuration precedence](configuration.md) — where a value actually comes from

```{toctree}
:hidden:

architecture
backends
swarmjob-lifecycle
watchdog-collector
configuration
```
