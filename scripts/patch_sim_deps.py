"""Patch installed sapien 3.0.0b1 / mplib 0.2.1 exactly as RoboTwin's scripts/_install.sh
and LeRobot's docker/Dockerfile.benchmark.robotwin do. Idempotent."""

import pathlib
import re
import site


def main():
    for d in site.getsitepackages():
        p = pathlib.Path(d) / "mplib" / "planner.py"
        if p.exists():
            s = p.read_text()
            n = re.sub(r"\bor collide\b", "", s, count=1)  # broken collision clause in screw planner
            p.write_text(n)
            print(f"mplib  {p}: {'patched' if s != n else 'already patched'}")
        p = pathlib.Path(d) / "sapien" / "wrapper" / "urdf_loader.py"
        if p.exists():
            s = p.read_text()
            n = s.replace('"srdf"', '".srdf"')  # srdf extension check
            n = re.sub(r'(open\([^)]*?, "r")\)( as)', r'\1, encoding="utf-8")\2', n)  # utf-8 URDF/SRDF
            p.write_text(n)
            print(f"sapien {p}: {'patched' if s != n else 'already patched'}")


if __name__ == "__main__":
    main()
