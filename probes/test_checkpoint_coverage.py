"""A permissive upstream loader must not accept an untrained action expert."""
import unittest
from types import SimpleNamespace
from stack_probe_common import validate_checkpoint_state


class CheckpointCoverageTest(unittest.TestCase):
    def setUp(self):
        self.state = {"mixtures.action.weight": SimpleNamespace(shape=(2, 3)),
                      "mixtures.video.weight": SimpleNamespace(shape=(3, 4))}
        self.model = SimpleNamespace(mot=SimpleNamespace(state_dict=lambda: self.state), proprio_encoder=None)

    def test_complete_state(self):
        result = validate_checkpoint_state(self.model, {"mot": self.state})
        self.assertTrue(result["verified"])
        self.assertEqual(result["action_keys"], 1)

    def test_video_only_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "video-only"):
            validate_checkpoint_state(self.model, {"dit": self.state})

    def test_missing_action_weights(self):
        with self.assertRaisesRegex(ValueError, "1 missing"):
            validate_checkpoint_state(self.model, {"mot": {"mixtures.video.weight": self.state["mixtures.video.weight"]}})

    def test_wrong_shape(self):
        stored = {**self.state, "mixtures.action.weight": SimpleNamespace(shape=(3, 2))}
        with self.assertRaisesRegex(ValueError, "1 wrong shapes"):
            validate_checkpoint_state(self.model, {"mot": stored})

    def test_missing_proprioception(self):
        self.model.proprio_encoder = SimpleNamespace(state_dict=lambda: {"weight": SimpleNamespace(shape=(2, 2))})
        with self.assertRaisesRegex(ValueError, "proprioception"):
            validate_checkpoint_state(self.model, {"mot": self.state})


if __name__ == "__main__":
    unittest.main()
