import io
import json
import unittest

from scientific_calculus.cli import run


class CliTest(unittest.TestCase):
    def test_json_end_to_end(self) -> None:
        payload = {
            "faces": [
                {"id": "g", "centroid": [0, 0]},
                {"id": "j", "centroid": [1, 0]},
                {"id": "a", "centroid": [2, 1]},
                {"id": "b", "centroid": [2, -1]},
            ],
            "adjacencies": [["g", "j"], ["j", "a"], ["j", "b"]],
            "goal": "g",
            "starts": ["a", "b"],
            "config": {
                "random_restarts": 4,
                "local_iterations": 10,
                "seed": 7,
            },
        }
        standard_output = io.StringIO()
        standard_error = io.StringIO()

        exit_code = run(
            [],
            standard_input=io.StringIO(json.dumps(payload)),
            standard_output=standard_output,
            standard_error=standard_error,
        )
        result = json.loads(standard_output.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(standard_error.getvalue(), "")
        self.assertEqual(result["cost"], 5.0)
        self.assertEqual(result["nodes"], ["a", "b", "g", "j"])
        self.assertEqual(
            result["edges"],
            [
                {"parent": "g", "child": "j", "weight": 1.0},
                {"parent": "j", "child": "a", "weight": 2.0},
                {"parent": "j", "child": "b", "weight": 2.0},
            ],
        )

    def test_invalid_input_returns_nonzero_with_message(self) -> None:
        standard_error = io.StringIO()

        exit_code = run(
            [],
            standard_input=io.StringIO("{}"),
            standard_output=io.StringIO(),
            standard_error=standard_error,
        )

        self.assertEqual(exit_code, 2)
        self.assertIn("faces", standard_error.getvalue())


if __name__ == "__main__":
    unittest.main()
