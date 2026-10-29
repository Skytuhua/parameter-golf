"""Check the evaluator's integer indexing without importing torch or training."""

import ast
from pathlib import Path
import unittest


SOURCE = Path(__file__).with_name("train_gpt.py")
FUNCTION = next(
    node for node in ast.parse(SOURCE.read_text(encoding="utf-8")).body
    if isinstance(node, ast.FunctionDef) and node.name == "eval_val_sliding"
)
EXPRESSIONS = {}
for node in ast.walk(FUNCTION):
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        EXPRESSIONS.setdefault(
            node.targets[0].id, compile(ast.Expression(node.value), str(SOURCE), "eval")
        )


def scored_windows(total_tokens, seq_len, stride, rank=0, world_size=1):
    state = dict(total_tokens=total_tokens, seq_len=seq_len, stride=stride,
                 rank=rank, world_size=world_size)
    for name in ("window_starts", "total_windows", "my_s", "my_e", "my_windows"):
        state[name] = eval(EXPRESSIONS[name], state)
    result = []
    for ws in state["my_windows"]:
        state["ws"] = ws
        for name in ("end", "wlen", "s"):
            state[name] = eval(EXPRESSIONS[name], state)
        result.append((ws, list(range(ws + state["s"], ws + state["wlen"]))))
    return result


class SlidingCoverageTest(unittest.TestCase):
    def test_hand_checked_windows(self):
        # Indices are targets in val_tokens[1:], so the first window scores 0..3.
        cases = [
            (1, 2, [(0, [0])]),
            (3, 2, [(0, [0, 1, 2])]),
            (4, 2, [(0, [0, 1, 2, 3])]),
            (6, 2, [(0, [0, 1, 2, 3]), (2, [4, 5])]),
            (9, 2, [(0, [0, 1, 2, 3]), (2, [4, 5]), (4, [6, 7]), (6, [8])]),
            (10, 2, [(0, [0, 1, 2, 3]), (2, [4, 5]), (4, [6, 7]), (6, [8, 9])]),
            (11, 3, [(0, [0, 1, 2, 3]), (3, [4, 5, 6]), (6, [7, 8, 9]), (9, [10])]),
        ]
        for total_tokens, stride, expected in cases:
            with self.subTest(total_tokens=total_tokens, stride=stride):
                self.assertEqual(scored_windows(total_tokens, 4, stride), expected)

    def test_rank_partitions(self):
        expected = [
            [(0, [0, 1, 2, 3])], [(2, [4, 5])], [(4, [6, 7])], [(6, [8])],
        ]
        for rank in range(4):
            self.assertEqual(scored_windows(9, 4, 2, rank, 4), expected[rank])
        # More ranks than windows: empty ranks must not repeat another rank's prefix.
        expected = [[], expected[0], [], expected[1], [], expected[2], [], expected[3]]
        for rank in range(8):
            self.assertEqual(scored_windows(9, 4, 2, rank, 8), expected[rank])

    def test_once_per_target_across_strides_and_ranks(self):
        for total_tokens in range(1, 25):
            for seq_len in range(2, 9):
                for stride in range(1, seq_len + 1):
                    for world_size in (1, 2, 4, 8):
                        with self.subTest(tokens=total_tokens, length=seq_len,
                                          stride=stride, ranks=world_size):
                            windows = [window for rank in range(world_size)
                                       for window in scored_windows(
                                           total_tokens, seq_len, stride, rank, world_size)]
                            self.assertTrue(all(targets for _, targets in windows))
                            self.assertEqual([t for _, targets in windows for t in targets],
                                             list(range(total_tokens)))


if __name__ == "__main__":
    unittest.main()
