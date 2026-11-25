"""Fill ${VARIABLES} in a manifest from the environment (no envsubst needed).

    COLLECTION=rahbar_candidate EMBED_MODEL=nomic-embed-text python3 scripts/render.py \\
        k8s/candidate.yaml | kubectl apply -f -

A variable that is not set is an error, so a Job is never created with an empty value.
"""

from __future__ import annotations

import os
import re
import sys


def render(text: str, env: dict[str, str]) -> str:
    missing = sorted({m for m in re.findall(r"\$\{([A-Z_][A-Z0-9_]*)\}", text) if m not in env})
    if missing:
        raise KeyError(", ".join(missing))
    return re.sub(r"\$\{([A-Z_][A-Z0-9_]*)\}", lambda m: env[m.group(1)], text)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: render.py <manifest.yaml>", file=sys.stderr)
        return 2
    try:
        with open(args[0], encoding="utf-8") as fh:
            sys.stdout.write(render(fh.read(), dict(os.environ)))
    except KeyError as exc:
        print(f"render: set these variables first: {exc.args[0]}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"render: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
