"""Tests using installed libraries, but no model weights or network access."""
import importlib.util
import unittest
from types import SimpleNamespace


HAS_DEPENDENCIES = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "gradio"))


@unittest.skipUnless(HAS_DEPENDENCIES, "Install app dependencies for integration tests")
class IntegrationTests(unittest.TestCase):
    def test_chunk_generation_preserves_order_and_integer_ids(self):
        import torch
        from app import generate_chunks

        calls = []

        def generate(**chunk):
            calls.append(chunk)
            self.assertEqual(chunk["input_features"].shape[0], 1)
            self.assertEqual(chunk["input_features"].dtype, torch.float32)
            self.assertEqual(chunk["decoder_input_ids"].dtype, torch.int64)
            self.assertNotIn("audio_chunk_index", chunk)
            return torch.tensor([[5, 6]]) if len(calls) == 1 else torch.tensor([[7]])

        model = SimpleNamespace(device="cpu", dtype=torch.float32, config=SimpleNamespace(pad_token_id=0), generate=generate)
        inputs = {
            "input_features": torch.arange(12, dtype=torch.float64).reshape(2, 2, 3),
            "decoder_input_ids": torch.tensor([[1, 2], [3, 4]]),
            "audio_chunk_index": [(0, 0), (0, 1)],
        }
        output = generate_chunks(model, inputs, lambda *args, **kwargs: None)
        self.assertEqual(output.tolist(), [[5, 6], [7, 0]])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["decoder_input_ids"].tolist(), [[3, 4]])
        self.assertEqual(inputs["input_features"].dtype, torch.float64)

    def test_ui_serializes_all_model_callbacks(self):
        from app import create_demo

        demo = create_demo()
        callbacks = [fn for fn in demo.fns.values() if fn.fn]
        self.assertEqual(len(callbacks), 3)
        self.assertTrue(all(fn.concurrency_id == "model" and fn.concurrency_limit == 1 for fn in callbacks))
        demo.close()


if __name__ == "__main__":
    unittest.main()
