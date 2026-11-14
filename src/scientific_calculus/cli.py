"""JSON 入出力のコマンドラインインターフェース。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, TextIO

from .io import parse_problem, solution_to_dict
from .solver import solve


def _read_json(path: str, standard_input: TextIO) -> object:
    if path == "-":
        return json.load(standard_input)
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path: str, payload: object, standard_output: TextIO) -> None:
    if path == "-":
        json.dump(payload, standard_output, ensure_ascii=False, indent=2)
        standard_output.write("\n")
        return
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="steiner-connect",
        description="面隣接グラフ上でスタートとゴールを結ぶ共有経路木を近似します。",
    )
    parser.add_argument(
        "input", nargs="?", default="-", help="入力JSONファイル（既定: 標準入力）"
    )
    parser.add_argument(
        "-o", "--output", default="-", help="出力JSONファイル（既定: 標準出力）"
    )
    return parser


def run(
    argv: Sequence[str] | None = None,
    *,
    standard_input: TextIO = sys.stdin,
    standard_output: TextIO = sys.stdout,
    standard_error: TextIO = sys.stderr,
) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        payload = _read_json(arguments.input, standard_input)
        graph, goal, starts, config = parse_problem(payload)
        solution = solve(graph, goal, starts, config)
        _write_json(
            arguments.output, solution_to_dict(graph, solution), standard_output
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        standard_error.write(f"error: {error}\n")
        return 2
    return 0


def main() -> None:
    raise SystemExit(run())
