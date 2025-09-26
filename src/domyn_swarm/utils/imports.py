try:
    from leptonai.api.v2.client import APIClient as _LeptonAPIClient

    _HAVE_LEPTON = True
    _ = _LeptonAPIClient()
except Exception as e:  # pylint: disable=broad-except
    print(f"Lepton import failed: {e}")
    _LeptonAPIClient = None  # type: ignore
    _HAVE_LEPTON = False


def _require_lepton():
    if not _HAVE_LEPTON:
        raise ImportError("Install `domyn-swarm[lepton]` to use the Lepton backend.")
