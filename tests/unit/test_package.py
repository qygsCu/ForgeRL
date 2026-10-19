"""Smoke tests for the installable package."""


def test_package_is_importable() -> None:
    """The installed top-level package can be imported."""
    import forgerl

    assert forgerl.__name__ == "forgerl"
