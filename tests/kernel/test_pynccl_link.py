import unittest

from minisgl.kernel.pynccl import _find_nccl_library, _nccl_linker_flags


class NCCLLinkerFlagsTest(unittest.TestCase):
    def test_finds_installed_nccl_runtime(self) -> None:
        library = _find_nccl_library()

        self.assertIsNotNone(library)
        assert library is not None
        self.assertTrue(library.is_file())
        self.assertIn(library.name, ("libnccl.so", "libnccl.so.2"))

    def test_runtime_library_has_rpath(self) -> None:
        library = _find_nccl_library()
        assert library is not None

        self.assertEqual(
            _nccl_linker_flags(),
            [str(library), f"-Wl,-rpath,{library.parent}"],
        )


if __name__ == "__main__":
    unittest.main()
