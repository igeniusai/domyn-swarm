# Checkpointing and resuming

:::{note}
This guide is still being written. Until it lands, the flags are documented in the [CLI reference](../reference/cli.md) and summarised in [Submitting jobs](submitting-jobs.md#checkpointing).
:::

Checkpointing lets a failed run resume instead of starting over. The relevant flags are `--checkpoint-dir`, `--checkpoint-tag`, `--checkpoint-interval`, `--no-resume`, `--no-checkpointing` and `--global-resume`.
