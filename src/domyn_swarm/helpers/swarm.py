import uuid


def generate_swarm_name(name: str) -> str:
    """Generate a unique swarm name by appending a short UUID to the given name."""
    unique_id = uuid.uuid4()
    short_id = str(unique_id)[:8]
    return f"{name}-{short_id}"
