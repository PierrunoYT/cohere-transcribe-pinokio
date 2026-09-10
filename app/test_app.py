"""Fast regression tests; no model download or third-party packages required."""
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch


class AppTests(unittest.TestCase):
    def setUp(self):
        self.torch = MagicMock()
        self.torch.cuda.is_available.return_value = False
        self.transformers = MagicMock()
        self.gradio = MagicMock()
        self.gradio.Error = ValueError
        modules = {
            "torch": self.torch,
            "gradio": self.gradio,
            "transformers": self.transformers,
            "transformers.audio_utils": MagicMock(),
        }
        spec = importlib.util.spec_from_file_location("transcribe_app", Path(__file__).with_name("app.py"))
        self.app = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(self.app)

    def test_cpu_uses_float32(self):
        with patch.dict(self.app.os.environ, {}, clear=True):
            self.app.get_model()
        kwargs = self.transformers.CohereAsrForConditionalGeneration.from_pretrained.call_args.kwargs
        self.assertEqual(kwargs["device_map"], "cpu")
        self.assertIs(kwargs["dtype"], self.torch.float32)

    def test_concurrent_tokens_reuse_one_model(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda token: self.app.get_model("cpu", token), ["a", "b", None, "a"]))
        self.assertTrue(all(result == results[0] for result in results))
        self.transformers.CohereAsrForConditionalGeneration.from_pretrained.assert_called_once()

    def test_device_is_part_of_cache_key(self):
        self.app.get_model("cpu")
        self.app.get_model("cuda")
        self.assertEqual(self.transformers.CohereAsrForConditionalGeneration.from_pretrained.call_count, 2)

    def test_failed_load_can_be_retried(self):
        loader = self.transformers.CohereAsrForConditionalGeneration.from_pretrained
        loader.side_effect = [RuntimeError("download failed"), MagicMock()]
        with self.assertRaises(RuntimeError):
            self.app.get_model("cpu")
        self.app.get_model("cpu")
        self.assertEqual(loader.call_count, 2)

    def test_missing_invalid_and_empty_audio_do_not_load_model(self):
        self.app.get_model = MagicMock()
        self.assertEqual(self.app.transcribe_audio(None, "English", True, None)[1], "")
        with self.assertRaisesRegex(ValueError, "supported language"):
            self.app.transcribe_audio("audio.wav", "Unknown", True, None)
        self.app.load_audio.return_value = []
        with self.assertRaisesRegex(ValueError, "empty"):
            self.app.transcribe_audio("audio.wav", "English", True, None)
        self.app.get_model.assert_not_called()

    def test_both_tabs_return_reassembled_string(self):
        processor = MagicMock()
        processor.return_value.get.return_value = [(0, 0), (0, 1)]
        processor.decode.return_value = ["complete transcript"]
        model = SimpleNamespace(device="cpu", dtype="float32", generate=MagicMock())
        self.app.get_model = MagicMock(return_value=(processor, model))
        self.app.generate_chunks = MagicMock()
        self.app.load_audio.return_value = [0] * 16000
        for callback in (self.app.transcribe_audio, self.app.transcribe_long_audio):
            text, stats = callback("audio.wav", "Japanese", False, " token ")
            self.assertEqual(text, "complete transcript")
            self.assertIn("Audio duration:", stats)
            self.assertEqual(processor.decode.call_args.kwargs["language"], "ja")
            self.assertEqual(processor.decode.call_args.kwargs["audio_chunk_index"], [(0, 0), (0, 1)])
            self.app.get_model.assert_called_with(hf_token="token")


if __name__ == "__main__":
    unittest.main()
