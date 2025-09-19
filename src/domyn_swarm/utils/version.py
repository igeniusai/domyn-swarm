import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def get_version() -> str:
    try:
        return version("domyn-swarm")
    except PackageNotFoundError:
        # Fallback when running from source without an installed dist
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib  # type: ignore[import-not-found]
        data = tomllib.loads(Path("pyproject.toml").read_text())
        return (data.get("project") or {}).get("version", "0.0.0+unknown")


if __name__ == "__main__":
    print(get_version())
