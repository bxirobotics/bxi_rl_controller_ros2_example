import unittest
import warnings

from bxi_example_py_elf3.utils.mod_system import _resolve_state_manifest_indexes


def _state(index=None):
    manifest = {} if index is None else {"index": index}
    return {"manifest": manifest}


class ResolveStateManifestIndexesTest(unittest.TestCase):
    def test_conflicts_are_reassigned_without_cascading(self):
        states = {
            "example/first": _state(1),
            "example/conflict": _state(1),
            "example/reserved": _state(2),
            "example/second_conflict": _state(1),
            "example/unindexed": _state(),
        }

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _resolve_state_manifest_indexes(states)

        self.assertEqual(states["example/first"]["manifest"]["index"], 1)
        self.assertEqual(states["example/conflict"]["manifest"]["index"], 3)
        self.assertEqual(states["example/reserved"]["manifest"]["index"], 2)
        self.assertEqual(
            states["example/second_conflict"]["manifest"]["index"], 4
        )
        self.assertNotIn("index", states["example/unindexed"]["manifest"])
        self.assertEqual(len(caught), 2)
        self.assertTrue(all(item.category is RuntimeWarning for item in caught))

    def test_invalid_index_still_fails(self):
        for index in (-1, True, "1"):
            with self.subTest(index=index):
                with self.assertRaisesRegex(
                    ValueError, "must be a non-negative integer"
                ):
                    _resolve_state_manifest_indexes(
                        {"example/invalid": _state(index)}
                    )


if __name__ == "__main__":
    unittest.main()
