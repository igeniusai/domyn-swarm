# Environment variables

`Settings` centralises configuration read from environment variables and
optional `.env` files. It reads, in order:

- `.env` in the current working directory
- `~/.domyn_swarm/.env`

Variables use the prefix `DOMYN_SWARM_` (case-insensitive) **unless the field
declares an explicit alias** — the Field column below shows the alias where one
exists. Values are parsed and validated by Pydantic.

```{include} ../_generated/settings-table.md
```
