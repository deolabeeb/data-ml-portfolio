"""Print non-sensitive local setup diagnostics. Does not install or change anything."""
import importlib.metadata as metadata
import json
import platform
import sys

packages = {}
for name in ("fastapi", "pydantic", "uvicorn", "pytest", "httpx"):
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = "not installed in this interpreter"

report = {
    "python": platform.python_version(),
    "supported_python": (3, 11) <= sys.version_info[:2] < (3, 14),
    "platform": platform.system(),
    "packages": packages,
    "api_key_needed": False,
    "docker_needed_for_M0": False,
    "note": "No environment variables, tokens, usernames, or credentials are printed.",
}
print(json.dumps(report, indent=2))
if not report["supported_python"]:
    raise SystemExit("Use Python 3.11, 3.12, or 3.13 for this starter.")
