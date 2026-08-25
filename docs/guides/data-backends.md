# Choosing a data backend

:::{note}
This guide is still being written. Until it lands, see [Installation](../getting-started/installation.md#optional-extras) for what each extra provides.
:::

A *data* backend decides how job input and output are read, written and executed — unrelated to the serving and compute backends in [Serving vs compute backends](../concepts/backends.md). The options are `pandas` (default, always available), `polars` and `ray`, selected with `--data-backend`.
