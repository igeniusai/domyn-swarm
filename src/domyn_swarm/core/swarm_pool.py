from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any, Generator, cast

from domyn_swarm import DomynLLMSwarm
from domyn_swarm.models.swarm import DomynLLMSwarmConfig


@contextmanager
def create_swarm_pool(
    *configs_or_swarms: "DomynLLMSwarmConfig | DomynLLMSwarm",
    config_or_swarm_list: list[DomynLLMSwarmConfig] | list[DomynLLMSwarm] | None = None,
    max_workers=None,
) -> Generator[tuple["DomynLLMSwarm"], Any, None]:
    """
    You can use this utility function like this:

    ```
    with create_swarm_pool(cfg1, cfg2, cfg3, max_workers=3) as (sw1, sw2, sw3):
        sw1.submit_job()
        sw2.submit_job()
        sw3.submit_job()
    ```

    or

    ```
    with create_swarm_pool(*my_cfg_list, max_workers=5) as swarms:
        for swarm in swarms:
            swarm.submit_job()
    ```

    """

    # 1) instantiate all the context‐manager objects
    # Casting required by pyright to understand the type
    if configs_or_swarms and isinstance(configs_or_swarms[0], DomynLLMSwarmConfig):
        configs = cast(list[DomynLLMSwarmConfig], configs_or_swarms)
        cms = [DomynLLMSwarm(cfg=config) for config in configs]
    elif configs_or_swarms and isinstance(configs_or_swarms[0], DomynLLMSwarm):
        cms = cast(list[DomynLLMSwarm], configs_or_swarms)
    else:
        raise ValueError(
            "configs_or_swarms must be either a sequence of DomynLLMSwarmConfig or DomynLLMSwarm"
        )

    entered = []
    try:
        # 2) call each __enter__ in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            # submit each cm.__enter__(); collect futures keyed by cm
            futures = {exe.submit(cm.__enter__): cm for cm in cms}
            for future in as_completed(futures):
                cm = futures[future]
                res = future.result()  # will re‐raise if __enter__ failed
                entered.append((cm, res))

        # 3) yield the tuple of entered results in the original order
        #    (filter out any that didn’t make it into 'entered' if one failed)
        #    Note: if one __enter__ raised, we jump straight to finally, cleaning up
        yield tuple(res for _, res in entered)

    finally:
        # 4) tear them all down, in reverse order of successful entry
        #    pass None for exc_type, exc_val, tb if no exception
        for cm, _ in reversed(entered):
            try:
                cm.__exit__(None, None, None)
            except Exception:
                pass
