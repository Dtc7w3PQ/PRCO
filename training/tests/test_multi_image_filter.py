import contextlib
import io
import unittest
from unittest.mock import patch

from datasets import Dataset

from verl.utils.dataset import RLHFDataset


class MultiImageFilterTests(unittest.TestCase):
    def test_skips_multi_image_samples_before_prompt_processing(self):
        for filter_prompts in (False, True):
            with self.subTest(filter_prompts=filter_prompts):
                source = Dataset.from_list(
                    [
                        {"problem": "first", "pictures": ["first.png"]},
                        {"problem": "multiple", "pictures": ["missing_a.png", "missing_b.png"]},
                        {"problem": "last", "pictures": ["last.png"]},
                    ]
                )
                checked = []

                def check_prompt(dataset_instance, example):
                    if len(example["pictures"]) != 1:
                        raise AssertionError("Multi-image sample reached prompt processing")
                    checked.append(example["problem"])
                    return True

                output = io.StringIO()
                errors = io.StringIO()
                with (
                    patch("verl.utils.dataset.load_dataset", return_value=source),
                    patch.object(RLHFDataset, "_filter_overlong_prompts", new=check_prompt),
                    contextlib.redirect_stdout(output),
                    contextlib.redirect_stderr(errors),
                ):
                    dataset = RLHFDataset(
                        data_path="test-dataset",
                        tokenizer=None,
                        processor=None,
                        image_key="pictures",
                        filter_overlong_prompts=filter_prompts,
                        filter_overlong_prompts_workers=None,
                    )
                self.assertEqual(list(dataset.dataset["problem"]), ["first", "last"])
                self.assertEqual(checked, ["first", "last"] if filter_prompts else [])
                self.assertEqual(len(source), 3)
                self.assertEqual(output.getvalue(), "")
                if not filter_prompts:
                    self.assertEqual(errors.getvalue(), "")
                self.assertNotIn("multi-image", errors.getvalue())

    def test_all_multi_image_samples_report_empty_dataset(self):
        source = Dataset.from_list([{"images": ["first.png", "second.png"]}])
        with patch("verl.utils.dataset.load_dataset", return_value=source):
            with self.assertRaisesRegex(ValueError, "No samples remain.*multi-image"):
                RLHFDataset("test-dataset", tokenizer=None, processor=None, filter_overlong_prompts=False)

    def test_keeps_samples_without_multiple_images(self):
        for records in (
            [{"images": ["single.png"]}, {"images": []}, {"images": None}],
            [{"problem": "text only"}],
        ):
            with self.subTest(records=records):
                source = Dataset.from_list(records)
                with patch("verl.utils.dataset.load_dataset", return_value=source):
                    dataset = RLHFDataset(
                        "test-dataset", tokenizer=None, processor=None, filter_overlong_prompts=False
                    )
                self.assertEqual(dataset.dataset.to_list(), records)


if __name__ == "__main__":
    unittest.main()
