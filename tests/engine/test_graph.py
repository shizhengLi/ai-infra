import unittest

from minisgl.engine.graph import _determine_cuda_graph_bs


class DetermineCudaGraphBatchSizesTest(unittest.TestCase):
    def determine(self, graph_max_bs: int | None, max_running_req: int) -> list[int]:
        return _determine_cuda_graph_bs(
            cuda_graph_bs=None,
            cuda_graph_max_bs=graph_max_bs,
            free_memory=40 * (1 << 30),
            max_running_req=max_running_req,
        )

    def test_respects_effective_max(self) -> None:
        cases = [
            (0, 128, []),
            (1, 128, [1]),
            (3, 128, [1, 2, 3]),
            (10, 128, [1, 2, 4, 8, 10]),
            (160, 10, [1, 2, 4, 8, 10]),
        ]
        for graph_max_bs, max_running_req, expected in cases:
            with self.subTest(graph_max_bs=graph_max_bs, max_running_req=max_running_req):
                self.assertEqual(self.determine(graph_max_bs, max_running_req), expected)

    def test_caps_auto_policy_to_concurrency(self) -> None:
        graph_bs = self.determine(graph_max_bs=None, max_running_req=128)

        self.assertEqual(graph_bs[-1], 128)
        self.assertTrue(all(bs <= 128 for bs in graph_bs))

    def test_normalizes_explicit_sizes(self) -> None:
        self.assertEqual(
            _determine_cuda_graph_bs(
                cuda_graph_bs=[8, 2, 8, -1, 256],
                cuda_graph_max_bs=None,
                free_memory=40 * (1 << 30),
                max_running_req=128,
            ),
            [2, 8],
        )


if __name__ == "__main__":
    unittest.main()
