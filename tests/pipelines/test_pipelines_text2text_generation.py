# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from transformers import (
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    Text2TextGenerationPipeline,
    pipeline,
)
from transformers.testing_utils import is_pipeline_test, require_tf, require_torch
from transformers.utils import is_torch_available

from .test_pipelines_common import ANY, PipelineTestCaseMeta


if is_torch_available():
    import torch


@is_pipeline_test
class Text2TextGenerationPipelineTests(unittest.TestCase, metaclass=PipelineTestCaseMeta):
    model_mapping = MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
    tf_model_mapping = TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING

    def get_test_pipeline(self, model, tokenizer, feature_extractor):
        generator = Text2TextGenerationPipeline(model=model, tokenizer=tokenizer)
        return generator, ["Something to write", "Something else"]

    def run_pipeline_test(self, generator, _):
        outputs = generator("Something there")
        self.assertEqual(outputs, [{"generated_text": ANY(str)}])
        # These are encoder decoder, they don't just append to incoming string
        self.assertFalse(outputs[0]["generated_text"].startswith("Something there"))

        outputs = generator(["This is great !", "Something else"], num_return_sequences=2, do_sample=True)
        self.assertEqual(
            outputs,
            [
                [{"generated_text": ANY(str)}, {"generated_text": ANY(str)}],
                [{"generated_text": ANY(str)}, {"generated_text": ANY(str)}],
            ],
        )

        outputs = generator(
            ["This is great !", "Something else"], num_return_sequences=2, batch_size=2, do_sample=True
        )
        self.assertEqual(
            outputs,
            [
                [{"generated_text": ANY(str)}, {"generated_text": ANY(str)}],
                [{"generated_text": ANY(str)}, {"generated_text": ANY(str)}],
            ],
        )

        with self.assertRaises(ValueError):
            generator(4)

    @require_torch
    def test_small_model_pt(self):
        generator = pipeline("text2text-generation", model="patrickvonplaten/t5-tiny-random", framework="pt")
        # do_sample=False necessary for reproducibility
        outputs = generator("Something there", do_sample=False)
        self.assertEqual(outputs, [{"generated_text": ""}])

        num_return_sequences = 3
        outputs = generator(
            "Something there",
            num_return_sequences=num_return_sequences,
            num_beams=num_return_sequences,
        )
        target_outputs = [
            {"generated_text": "Beide Beide Beide Beide Beide Beide Beide Beide Beide"},
            {"generated_text": "Beide Beide Beide Beide Beide Beide Beide Beide"},
            {"generated_text": ""},
        ]
        self.assertEqual(outputs, target_outputs)

        outputs = generator("This is a test", do_sample=True, num_return_sequences=2, return_tensors=True)
        self.assertEqual(
            outputs,
            [
                {"generated_token_ids": ANY(torch.Tensor)},
                {"generated_token_ids": ANY(torch.Tensor)},
            ],
        )
        generator.tokenizer.pad_token_id = generator.model.config.eos_token_id
        generator.tokenizer.pad_token = "<pad>"
        outputs = generator(
            ["This is a test", "This is a second test"],
            do_sample=True,
            num_return_sequences=2,
            batch_size=2,
            return_tensors=True,
        )
        self.assertEqual(
            outputs,
            [
                [
                    {"generated_token_ids": ANY(torch.Tensor)},
                    {"generated_token_ids": ANY(torch.Tensor)},
                ],
                [
                    {"generated_token_ids": ANY(torch.Tensor)},
                    {"generated_token_ids": ANY(torch.Tensor)},
                ],
            ],
        )

    @require_tf
    def test_small_model_tf(self):
        generator = pipeline("text2text-generation", model="patrickvonplaten/t5-tiny-random", framework="tf")
        # do_sample=False necessary for reproducibility
        outputs = generator("Something there", do_sample=False)
        self.assertEqual(outputs, [{"generated_text": ""}])
