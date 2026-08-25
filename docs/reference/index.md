# Reference

The CLI, configuration and environment-variable pages are generated from the
source on every build, so they cannot drift from the code you have installed.

- [CLI](cli.md) — every command and flag, from the Typer application
- [Configuration](configuration.md) — every YAML field, from the Pydantic models
- [Environment variables](environment.md) — every variable, from the `Settings` model
- [Python API](api/index.md) — the four public objects, curated
- [Release notes](changelog.md) — the changelog

The Python API pages are deliberately *not* generated over the whole package.
Only the names exported from `domyn_swarm` are documented; everything else is
internal and may change without notice.

```{toctree}
:hidden:

cli
configuration
environment
api/index
changelog
```
