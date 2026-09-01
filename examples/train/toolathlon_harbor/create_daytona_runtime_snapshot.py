"""Build the shared Toolathlon runtime remotely as a named Daytona snapshot."""

from __future__ import annotations

import argparse
import asyncio
import tarfile
import tempfile
from pathlib import Path


async def create_snapshot(archive: Path, name: str, target: str | None) -> None:
    from daytona import (
        AsyncDaytona,
        CreateSnapshotParams,
        DaytonaConfig,
        Image,
        Resources,
    )

    config = DaytonaConfig(target=target) if target else None
    daytona = AsyncDaytona(config) if config else AsyncDaytona()
    try:
        try:
            existing = await daytona.snapshot.get(name)
        except Exception as error:
            if (
                "not found" not in str(error).lower()
                and "notfound" not in type(error).__name__.lower()
            ):
                raise
        else:
            print(
                f"Snapshot {name!r} already exists in state {existing.state}; nothing to do"
            )
            return

        with tempfile.TemporaryDirectory(prefix="toolathlon-runtime-") as temporary:
            root = Path(temporary)
            with tarfile.open(archive, "r:gz") as bundle:
                bundle.extractall(root, filter="data")
            dockerfiles = list(root.glob("*/Dockerfile"))
            if len(dockerfiles) != 1:
                raise RuntimeError(
                    f"expected one runtime Dockerfile in {archive}, found {len(dockerfiles)}"
                )
            await daytona.snapshot.create(
                CreateSnapshotParams(
                    name=name,
                    image=Image.from_dockerfile(str(dockerfiles[0])),
                    resources=Resources(cpu=2, memory=4, disk=20),
                )
            )
        print(f"Submitted remote Daytona build for snapshot {name!r}")
    finally:
        await daytona.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--target", help="optional Daytona target, for example us")
    args = parser.parse_args()
    asyncio.run(create_snapshot(args.archive.resolve(), args.name, args.target))


if __name__ == "__main__":
    main()
