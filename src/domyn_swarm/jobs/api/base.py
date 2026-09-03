# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""
Light-weight framework for driver scripts that run **inside** a Domyn swarm.
Every class:

1.  Reads the load-balancer URL from the `ENDPOINT` env-var (injected by
    `DomynLLMSwarm` on the head node).
2.  Creates a single `openai.AsyncOpenAI` client pointing to that URL
    (`base_url=ENDPOINT`, `api_key="-"`).
3.  Provides `.run(df)` - a *synchronous* wrapper around an async
    coroutine so users don't have to think about `asyncio` unless they
    want to.
4.  Implements `.to_kwargs()` ⇒ JSON-serialisable dict so the object can
    be reconstructed by `domyn_swarm.jobs.cli.run` inside the allocation.

Sub-classes included:

* `CompletionJob`       → one prompt → one text completion
* `ChatCompletionJob`   → list-of-messages → one assistant reply
"""

import abc
from collections.abc import Awaitable, Callable
import difflib
import inspect
import logging
import os
from pathlib import Path
import threading
from typing import Any, ClassVar
import warnings

from deprecated import deprecated
from openai import AsyncOpenAI
import pandas as pd
from tqdm import tqdm

from domyn_swarm.checkpoint.manager import CheckpointManager
from domyn_swarm.config.settings import get_settings
from domyn_swarm.helpers.logger import setup_logger
from domyn_swarm.jobs.api.config import JobConfig, OutputJoinMode as OutputJoinMode

from .batching import BatchExecutor

logger = setup_logger(__name__, level=logging.INFO)


def _is_bare_value(value: Any) -> bool:
    """Return whether a class namespace entry is a plain configuration value.

    A subclass may define a `property` (or another descriptor) whose name
    matches a configuration field -- see `SwarmJob.__setattr__`. Such an entry
    is not a class-level configuration declaration and must not be swept into
    one, or `__init_subclass__` would delete the descriptor out from under the
    class.

    Args:
        value: The value found in a class's own `__dict__` for a field name.

    Returns:
        `False` for a `property`, `classmethod`, `staticmethod`, function, or
        other callable; `True` otherwise.
    """
    return not isinstance(value, property | classmethod | staticmethod) and not callable(value)


class SwarmJob(abc.ABC):
    """
    Abstract base class for distributed LLM processing jobs in the Domyn swarm framework.

    This class provides a robust foundation for running large-scale language model tasks with
    built-in reliability features including automatic checkpointing, retry mechanisms, and
    concurrent processing capabilities.

    Key Features:
        - **Automatic Checkpointing**: Periodically saves progress to enable recovery from failures
        - **Concurrent Processing**: Configurable parallelism with rate limiting
            and timeout handling
        - **Retry Logic**: Built-in exponential backoff for handling transient failures
        - **Provider Agnostic**: Supports multiple LLM providers (OpenAI, vLLM, etc.)
            via pluggable clients
        - **Callback System**: Extensible event hooks for monitoring and custom behavior
        - **DataFrame Integration**: Native pandas DataFrame support for batch processing

    Architecture:
        The class follows a template method pattern where subclasses implement the core
        `transform_items()` method while inheriting all reliability and concurrency infrastructure.
        Processing flows through: DataFrame → batching → transform_items → results → checkpointing.

    Configuration:
        Configuration lives on `JobConfig`, not on the job itself. A subclass
        declares it by setting a class-level `config` -- an instance of
        `JobConfig`, or of a subclass when the job adds fields of its own. No
        constructor is needed. Every configuration field is readable and
        writable as an attribute of the job (`self.threshold` below) and
        overridable at construction time
        (`MyJob(model="gpt-4", threshold=0.9)`). Provider request parameters
        (`temperature`, `top_p`, and the like) are not configuration fields:
        they live in `config.request_params` and are read back through
        `self.kwargs`.

    Example:
        ::

            class MyJobConfig(JobConfig):
                threshold: float = 0.5


            class MyJob(SwarmJob):
                config = MyJobConfig(
                    input_column_name="prompt", output_cols="answer", max_concurrency=8
                )

                async def transform_items(self, items: list[Any]) -> list[Any]:
                    # Process items using self.client
                    results = []
                    for item in items:
                        response = await self.client.chat.completions.create(
                            model=self.model, messages=item, **self.kwargs
                        )
                        results.append(response.choices[0].message.content)
                    return results


            job = MyJob(model="gpt-4")
            results_df = await job.run(input_df, tag="experiment_1")

    Attributes:
        config: The job's resolved configuration -- see `JobConfig`.
        client: Initialized async LLM client instance.
        results: Final processed DataFrame after job completion.

    Raises:
        RuntimeError: When ENDPOINT environment variable is missing
        ValueError: When required model name is not provided
        NotImplementedError: When required abstract methods are not implemented

    Note:
        Subclasses must implement the `transform_items()` method which processes a list of items
        and returns results in the same order. The framework handles all infrastructure concerns
        including error handling, checkpointing, progress tracking, and result aggregation.
        The `transform()` method is deprecated and should not be used.
    """

    api_version: int = 2

    config_class: ClassVar[type[JobConfig]] = JobConfig
    config: JobConfig

    def __init__(
        self,
        *,
        config: JobConfig | None = None,
        client=None,
        **overrides: Any,
    ):
        """Initialize the job from its configuration.

        Args:
            config: The job's configuration. Defaults to the class-level
                declaration, or to `config_class()` when there is none.
            client: A pre-built async LLM client. Not configuration, and not
                serialized.
            **overrides: Individual configuration fields, taking precedence over
                `config` and over any class-level declaration. A name that is not
                a configuration field is folded into `request_params` instead,
                with a `DeprecationWarning`, unless it closely resembles a field
                name -- then it is treated as a typo and raises.

        Raises:
            RuntimeError: If no endpoint is given and `ENDPOINT` is unset.
            ValueError: If the model name is empty.
            TypeError: If an override is not a configuration field and closely
                resembles one -- almost certainly a typo rather than a genuine
                provider request parameter.
        """
        if config is not None and not isinstance(config, self.config_class):
            raise TypeError(
                f"{type(self).__name__} expects a {self.config_class.__name__}, got "
                f"{type(config).__name__}."
            )

        base = config if config is not None else self._class_config()
        fields = type(base).model_fields

        legacy_output_col = overrides.pop("output_column_name", None)
        if legacy_output_col is not None:
            if "output_cols" in overrides:
                warnings.warn(
                    "Both 'output_column_name' and 'output_cols' parameters are "
                    "provided. The 'output_column_name' parameter is deprecated and "
                    "will be ignored in favor of 'output_cols'.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    "The 'output_column_name' parameter is "
                    "deprecated and will be removed in a future version. "
                    "Use 'output_cols' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                overrides["output_cols"] = legacy_output_col

        nested = overrides.pop("kwargs", None)
        unknown = [k for k in overrides if k not in fields]

        # A name that closely resembles a field is a typo, not a provider
        # parameter: routing it to request_params would silently leave the field
        # at its default. Only inexact names reach here, so an exact field name
        # can never be reported as a misspelling of itself.
        #
        # 0.75 rather than 0.8 so that 'retrys' (an ordinary typo of 'retries',
        # ratio 0.769) is caught; checked against the common provider parameter
        # names for false positives, which score well below it.
        NEAR_MISS_CUTOFF = 0.75
        for name in unknown:
            close = difflib.get_close_matches(name, fields, n=1, cutoff=NEAR_MISS_CUTOFF)
            if close:
                raise TypeError(
                    f"{name!r} is not a configuration field -- did you mean "
                    f"{close[0]!r}? If it really is a provider request parameter, "
                    f"pass request_params={{{name!r}: ...}}."
                )

        legacy_request_params = {k: overrides.pop(k) for k in unknown}
        if isinstance(nested, dict):
            legacy_request_params.update(nested)

        if legacy_request_params:
            # A fixed cutoff always has a tail of typos just below it, so the
            # warning names the closest field for anything loosely similar. Well
            # short of the raise threshold, and above the score common provider
            # parameter names reach against any field.
            HINT_CUTOFF = 0.6
            hints = {
                name: match[0]
                for name in unknown
                if (match := difflib.get_close_matches(name, fields, n=1, cutoff=HINT_CUTOFF))
            }
            hint = ""
            if hints:
                resemblances = ", ".join(f"{n!r} resembles {f!r}" for n, f in sorted(hints.items()))
                hint = f" ({resemblances})"
            warnings.warn(
                f"Passing provider request parameters as constructor arguments is "
                f"deprecated: {sorted(legacy_request_params)}{hint}. Pass "
                f"request_params={{...}} instead. This will be an error in "
                f"domyn-swarm 0.33.",
                DeprecationWarning,
                stacklevel=2,
            )
            # An explicit request_params={...} is the recommended, non-deprecated
            # form, so it takes precedence over the deprecated bare-kwarg form
            # when both name the same parameter.
            merged_request_params = {
                **base.request_params,
                **legacy_request_params,
                **overrides.pop("request_params", {}),
            }
            overrides["request_params"] = merged_request_params

        # Always a fresh config: the resolution above writes to it, and neither
        # a caller's config nor a class-level declaration may be mutated -- deep,
        # because a job that adds a provider parameter does it by mutating
        # `request_params` in place.
        self.config = base.merged_with(**overrides) if overrides else base.model_copy(deep=True)

        if not self.config.name:
            self.config.name = self.__class__.__name__
        if not self.config.endpoint:
            self.config.endpoint = os.getenv("ENDPOINT")
        if not self.config.endpoint:
            raise RuntimeError("ENDPOINT environment variable is not set")
        if not self.config.model:
            raise ValueError("Model name must be specified")

        headers = {}
        token = get_settings().resolved_api_token
        if token:
            logger.info("Using API_TOKEN from environment for authentication")
            headers["Authorization"] = f"Bearer {token.get_secret_value()}"

        self.client = client or AsyncOpenAI(
            base_url=f"{self.config.endpoint}/v1",
            api_key="-",
            organization="-",
            project="-",
            timeout=self.config.timeout,
            default_headers=headers,
            **(self.config.client_kwargs or {}),
        )
        self._callbacks: dict[str, Callable] = {}

        self.results = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Fold class-level configuration declarations into the class config.

        A bare class attribute whose name matches a configuration field is a
        declaration of that field's value. One that differs from the field
        default warns, because such declarations were silently ignored before
        domyn-swarm 0.32 and the change in behaviour is otherwise invisible.

        `config_class` follows `config`: whenever `type(config)` is a strict
        subclass of the `config_class` that would otherwise apply -- declared
        on this class or inherited -- `type(config)` is adopted instead, so
        declaring `config` alone is enough. A broader `config_class` declared
        explicitly is no reason to keep it, since `isinstance(config,
        config_class)` holds either way.

        The consequences of a `config_class` broader than `config` reach past
        a skipped validator. It is what `_class_config` rebuilds through, what
        validates constructor overrides, and what reconstructs the job on the
        far side of the wire, so a field only the more specific type defines
        would be rejected as an unrecognized override and lost on
        reconstruction.

        The two types must be related. If neither `type(config)` nor the
        applicable `config_class` is a subclass of the other, no type can
        carry every field `config` set, and the declaration raises rather
        than guessing.

        Args:
            **kwargs: Forwarded to `super().__init_subclass__`.

        Raises:
            TypeError: If a bare attribute and this same class's `config`
                object both declare a value for the same field. Declaring
                `config` for some fields and bare attributes for others is
                fine, as is a subclass overriding, via a bare attribute, a
                field an ancestor's `config` object set -- only a same-class,
                same-field conflict is contradictory.
            TypeError: If this class's `config` object's type and the
                `config_class` that would apply to it are unrelated types --
                neither a subclass of the other.
        """
        super().__init_subclass__(**kwargs)

        base = cls.__dict__.get("config")
        own_config_class = cls.__dict__.get("config_class")

        if isinstance(base, JobConfig):
            effective_config_class = own_config_class or cls.config_class
            config_is_more_specific = issubclass(type(base), effective_config_class)
            compatible = config_is_more_specific or issubclass(effective_config_class, type(base))
            if not compatible:
                raise TypeError(
                    f"{cls.__name__} declares config_class={effective_config_class.__name__} "
                    f"but its `config` object is a {type(base).__name__}, and neither is a "
                    f"subclass of the other -- reconstructing through "
                    f"{effective_config_class.__name__} would lose or misinterpret fields "
                    f"{type(base).__name__} actually set. Change `config` to a "
                    f"{effective_config_class.__name__} instance, or change `config_class` to "
                    f"{type(base).__name__} (or a shared base of the two)."
                )
            if config_is_more_specific:
                # `_class_config()` rebuilds through `config_class`, so it has
                # to name the most specific type available or the subclass's
                # validators and extra fields are dropped.
                cls.config_class = type(base)

        fields = cls.config_class.model_fields
        declared = {
            name: cls.__dict__[name]
            for name in fields
            if name in cls.__dict__ and _is_bare_value(cls.__dict__[name])
        }

        if isinstance(base, JobConfig):
            clash = sorted(set(declared) & base.model_fields_set)
            if clash:
                raise TypeError(
                    f"{cls.__name__} declares {clash} as both a bare class attribute "
                    f"and a field of its `config` object -- it is ambiguous which one "
                    f"should apply. Set each of {clash} in only one of the two places."
                )

        if not declared:
            return

        changed = sorted(name for name, value in declared.items() if value != fields[name].default)
        if changed:
            warnings.warn(
                f"{cls.__name__} declares {changed} as bare class attributes with "
                f"values other than the field defaults. These were silently ignored "
                f"before domyn-swarm 0.32; now `{cls.__name__}(...)` instances use "
                f"them unless a constructor argument overrides them. If the job "
                f"relied on the previous (ignored) declaration, pass the value "
                f"explicitly as a constructor argument instead.",
                UserWarning,
                stacklevel=2,
            )

        for name in declared:
            delattr(cls, name)
        cls._declared_config = dict(declared)

    @classmethod
    def _class_config(cls) -> JobConfig:
        """Return the configuration declared across the class hierarchy.

        Each class in the MRO may contribute fields two ways: a `config`
        object (only the fields *it* explicitly set) and bare attributes
        collected into `_declared_config` by `__init_subclass__`. The walk goes
        base-to-derived, so a more derived class's declaration for a field wins
        over a less derived one's, including a bare attribute overriding a field
        an ancestor's `config` object set.

        Returns:
            A config combining every ancestor's declared fields.
        """
        merged: dict[str, Any] = {}
        for klass in reversed(cls.__mro__):
            own_config = klass.__dict__.get("config")
            if isinstance(own_config, JobConfig):
                merged.update(own_config.model_dump(exclude_unset=True))
            merged.update(klass.__dict__.get("_declared_config") or {})
        return cls.config_class(**merged)

    def __getattr__(self, name: str) -> Any:
        """Read configuration fields as attributes.

        Only reached for names not found normally, so instance state and methods
        are unaffected.

        Known limitation: an `AttributeError` raised inside a subclass descriptor
        for `name` (e.g. a `property` getter) is treated by normal attribute lookup
        as "not found" and falls through to here, so the reported error is "no
        attribute `name`" even though the real fault is inside the descriptor.

        Args:
            name: Attribute name.

        Returns:
            The configuration field's value.

        Raises:
            AttributeError: If `name` is not a configuration field.
        """
        config = self.__dict__.get("config")
        if config is not None and name in type(config).model_fields:
            return getattr(config, name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        """Write configuration fields through to the config.

        A subclass property named after a config field is a data descriptor, so a
        *read* of `name` reaches the property rather than `__getattr__`. The
        property check here keeps writes symmetric: without it a write would skip
        the property and land in the config, bypassing whatever the setter does.

        Known limitation: an attribute assigned before `self.config` exists (that
        is, before `super().__init__()` runs) has no config to route into and lands
        in `self.__dict__`. Once `config` is assigned, that instance entry
        permanently shadows the config field of the same name -- reads never reach
        the config again, and the value never appears in `to_kwargs()`'s payload.

        Args:
            name: Attribute name.
            value: Value to assign.
        """
        if isinstance(getattr(type(self), name, None), property):
            object.__setattr__(self, name, value)
            return
        config = self.__dict__.get("config")
        if config is not None and name in type(config).model_fields:
            setattr(config, name, value)
            return
        object.__setattr__(self, name, value)

    @property
    def kwargs(self) -> dict[str, Any]:
        """Provider request parameters.

        Returns:
            The configured `request_params`.
        """
        return self.config.request_params

    def register_callback(self, event: str, fn: Callable) -> None:
        """Register a named callback (e.g., 'on_batch_done')."""
        self._callbacks[event] = fn

    def get_callback(self, event: str) -> Callable | None:
        return self._callbacks.get(event)

    async def run(
        self,
        df: pd.DataFrame,
        tag: str,
        checkpoint_dir: str | Path = ".checkpoints",
    ) -> pd.DataFrame:
        """
        Run the job end-to-end with checkpointing support.
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"{self.__class__.__name__}_{tag}.parquet"
        manager = CheckpointManager(
            path,
            df,
            expected_output_cols=self.output_cols,
            input_col=self.input_column_name,
        )

        todo_df = manager.filter_todo()
        idx_map = todo_df.index.to_numpy()

        async def flush(out_list, new_ids):
            thread_name = threading.current_thread().name
            manager.flush(out_list, new_ids, self.output_cols, idx_map)
            tqdm.write(
                f"[{thread_name}] Checkpoint flushed {len(new_ids)} "
                f"rows, new total: {len(manager.done_df)}"
            )

        self.register_callback("on_batch_done", flush)

        try:
            items = todo_df[self.input_column_name].tolist()
            await self.batched(items, self._call_unit)
        finally:
            self._callbacks.clear()

        self.results = manager.finalize()
        return self.results

    async def batched(self, seq: list, fn: Callable) -> list:
        """
        Run a batched async pipeline over `seq` using `fn`.

        Supports retrying and invokes the 'on_batch_done' callback if registered.
        """
        executor = BatchExecutor(self.max_concurrency, self.checkpoint_interval, self.retries)
        return await executor.run(
            seq,
            fn,
            on_batch_done=self.get_callback("on_batch_done"),
            progress=True,
        )

    @deprecated(reason="Legacy transform(df) is no longer supported; implement transform_items.")
    async def transform(self, df: pd.DataFrame):
        raise RuntimeError(
            "transform(df) is no longer supported. Implement transform_items(items) instead."
        )

    def to_kwargs(self) -> dict:
        """Return the configuration to serialize into `--job-kwargs`.

        Most fields are emitted only when a caller actually set them, so a
        merely derived value is derived again on the far side instead of
        arriving as if it had been chosen. Two are unconditional: `name`, which
        `__init__` always assigns, and a non-empty `request_params`, which a job
        can add to by mutating `self.kwargs` in place -- a mutation
        `exclude_unset` cannot see, because it performs no assignment.
        `endpoint` and `model` are supplied separately by the runner, and the
        client is not serializable, so neither appears here.

        Returns:
            A JSON-serializable configuration payload.
        """
        payload = self.config.model_dump(mode="json", exclude_unset=True)
        if self.config.request_params and "request_params" not in payload:
            payload["request_params"] = self.config.model_dump(
                mode="json", include={"request_params"}
            )["request_params"]
        payload.pop("endpoint", None)
        payload.pop("model", None)
        return payload

    def _request_kwargs(self) -> dict:
        """Return the parameters forwarded to the provider on each request.

        Returns:
            The configured `request_params`.
        """
        return dict(self.config.request_params)

    async def _call_unit(self, item: Any) -> Any:
        """
        Bridge: run `transform_items` on a single element and return the single result.
        Ensures the contract (len(out) == 1).
        """
        out = self.transform_items([item])
        if inspect.isawaitable(out):
            out = await out
        if not isinstance(out, list) or len(out) != 1:
            raise RuntimeError(
                "transform_items(items) must return a list of the same length as `items`."
            )
        return out[0]

    @abc.abstractmethod
    async def transform_items(self, items: list[Any]) -> list[Any]:
        """Pure transform: items -> results (same order). No I/O or checkpointing."""
        raise NotImplementedError("Sub-classes must implement `transform_items`.")

    async def transform_streaming(
        self,
        items: list[Any],
        *,
        on_flush: Callable[[list[int], list[Any]], Awaitable[None]],
        checkpoint_every: int,
    ):
        """Run a streaming transform without retaining all outputs in memory.

        Args:
            items: Input items to process.
            on_flush: Callback invoked with `(idxs, outputs)` for flushed batches.
            checkpoint_every: Number of items between flushes.
        """
        executor = BatchExecutor(self.max_concurrency, checkpoint_every, self.retries)
        await executor.run_streaming(
            items,
            self._call_unit,
            on_batch_done=lambda outs, idxs: on_flush(idxs, outs),
            progress=True,
        )
