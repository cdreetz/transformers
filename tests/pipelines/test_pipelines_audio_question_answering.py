# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import numpy as np

from transformers import AutoProcessor, AutoModelForCausalLM, is_torch_available
from transformers.testing_utils import require_torch, slow
from transformers.utils import cached_property

from ..test_modeling_common import floats_tensor

if is_torch_available():
    import torch


class AudioQuestionAnsweringPipelineTests(unittest.TestCase):
    def get_test_pipeline(self, model_name=None, processor=None, tokenizer=None):
        audio = np.random.rand(1600)
        question = "What is in this audio?"

        if model_name is None:
            model = self.get_model()
            processor = self.get_processor()
            # The model in this test case is not a real model but a mock.
            # We can pass any test tokenizer here as long as we don't actually use it for inference.
            tokenizer = processor

        if tokenizer is None:
            return None, None, None, None, None, None, None

        from transformers import pipeline
        from transformers.pipelines.audio_question_answering import AudioQuestionAnsweringPipeline

        audio_qa = AudioQuestionAnsweringPipeline(model=model, processor=processor)

        return audio_qa, audio, question, model, processor, tokenizer

    @cached_property
    def mock_processor(self):
        if not is_torch_available():
            return None
        
        # Create a mock processor for testing
        class MockProcessor:
            def __init__(self):
                self.model_input_names = ["input_ids", "attention_mask"]

            def __call__(self, text=None, audios=None, return_tensors=None):
                batch_size = 1
                sequence_length = 64
                vocab_size = 1024
                hidden_size = 128
                encoder_seq_length = 128

                input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
                attention_mask = torch.ones_like(input_ids)

                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }

            def batch_decode(self, output_ids, **kwargs):
                return ["This is a mock audio transcription"]
            
        return MockProcessor()

    @cached_property
    def mock_model(self):
        if not is_torch_available():
            return None

        # Create a mock model for testing
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.main_input_name = "input_ids"
                self.generation_config = None
                self.config = None

            def __call__(self, **kwargs):
                batch_size = kwargs["input_ids"].shape[0]
                sequence_length = kwargs["input_ids"].shape[1]
                vocab_size = 1024

                logits = floats_tensor((batch_size, sequence_length, vocab_size))
                return {"logits": logits}
            
            def generate(self, **kwargs):
                batch_size = kwargs["input_ids"].shape[0]
                sequence_length = kwargs["input_ids"].shape[1]
                
                # Return sequence with additional tokens (simulating generation)
                return torch.randint(0, 1024, (batch_size, sequence_length + 10))
            
            def can_generate(self):
                return True

        return MockModel()

    def get_model(self):
        return self.mock_model
    
    def get_processor(self):
        return self.mock_processor

    @require_torch
    def test_small_model_pt(self):
        audio_qa_pipeline, audio, question, model, processor, tokenizer = self.get_test_pipeline()
        if audio_qa_pipeline is None:
            return

        # Simple test
        output = audio_qa_pipeline(question=question, audio=audio)
        self.assertEqual(output, {"answer": "This is a mock audio transcription"})

        # Test with top_k
        output = audio_qa_pipeline(question=question, audio=audio, top_k=2)
        self.assertEqual(len(output), 1)  # Should return a list with only one item for this mock

    @slow
    @require_torch
    def test_phi_multimodal(self):
        model_id = "microsoft/Phi-4-multimodal-instruct"
        
        # Skip if not running slow tests
        try:
            # Will throw error if model is not available
            model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        except Exception:
            self.skipTest(f"Model {model_id} not available for testing")
            
        from transformers import pipeline
        
        # Create a simple audio array for testing
        sample_rate = 16000
        audio_data = np.sin(2 * np.pi * 440 * np.arange(0, 1, 1/sample_rate))
        
        # Initialize pipeline with the smallest possible configuration
        pipeline = pipeline(
            "audio-question-answering",
            model=model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Basic functionality test
        question = "What frequency is this sine wave?"
        results = pipeline(question=question, audio=(audio_data, sample_rate))
        
        # Verify we get a valid response
        self.assertIn("answer", results)
        self.assertIsInstance(results["answer"], str)
        self.assertTrue(len(results["answer"]) > 0)