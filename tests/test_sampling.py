import unittest
from rpbench.core import Backend
from rpbench.sampling import validate_sampling


class SamplingTests(unittest.TestCase):
    def test_rejects_typo(self):
        with self.assertRaisesRegex(ValueError, "temprature"):
            validate_sampling("llamacpp-native", "completion", {"temprature": 0.8})

    def test_rejects_native_parameter_on_generic_chat(self):
        with self.assertRaisesRegex(ValueError, "dry_multiplier"):
            validate_sampling("llamacpp", "chat", {"dry_multiplier": 0.8})

    def test_llama_native_sampling_preserved(self):
        backend = Backend("llamacpp-native", "http://127.0.0.1:8080", "model", "completion")
        _, payload = backend.prepare([{"role": "user", "content": "hello"}],
                                     {"max_tokens": 128, "min_p": 0.05, "dry_multiplier": 0.5}, {})
        self.assertEqual(payload["n_predict"], 128)
        self.assertEqual(payload["min_p"], 0.05)
        self.assertEqual(payload["dry_multiplier"], 0.5)

    def test_kobold_native_sampling_preserved(self):
        backend = Backend("koboldcpp-native", "http://127.0.0.1:5001", "model", "completion")
        _, payload = backend.prepare([{"role": "user", "content": "hello"}],
                                     {"max_tokens": 128, "rep_pen": 1.08}, {})
        self.assertEqual(payload["max_length"], 128)
        self.assertEqual(payload["rep_pen"], 1.08)

    def test_invalid_max_tokens(self):
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            validate_sampling("mock", "chat", {"max_tokens": -1})


if __name__ == "__main__":
    unittest.main()
