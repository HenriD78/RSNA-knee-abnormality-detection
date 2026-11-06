"""Keep the default test run independent of private competition artifacts."""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-local-assets", action="store_true", default=False,
        help="Include tests requiring local data, weights, wheels or completed runs.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-local-assets"):
        return
    skip = pytest.mark.skip(reason="Requires local artifacts; opt in with --run-local-assets")
    for item in items:
        if "local_assets" in item.keywords:
            item.add_marker(skip)
