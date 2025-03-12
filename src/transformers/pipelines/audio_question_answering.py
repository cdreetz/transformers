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

import warnings
from typing import List, Union
import numpy as np
import os
import io
import requests
import soundfile as sf
from urllib.request import urlopen

from ..utils import add_end_docstrings, is_torch_available, logging
from .base import Pipeline, build_pipeline_init_args


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch
    from ..models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


@add_end_docstrings(build_pipeline_init_args(has_tokenizer=True, has_processor=True))
class AudioQuestionAnsweringPipeline(Pipeline):
    """
    Audio question answering pipeline using any model that accepts audio and text inputs. This pipeline is currently only
    available in PyTorch and requires a processor from a model like Phi-4-multimodal.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> audio_qa = pipeline(model="microsoft/Phi-4-multimodal-instruct", task="audio-question-answering")
    >>> audio_url = "https://upload.wikimedia.org/wikipedia/commons/b/b0/Barbara_Sahakian_BBC_Radio4_The_Life_Scientific_29_May_2012_b01j5j24.flac"
    >>> audio_qa(question="Transcribe the audio to text", audio=audio_url)
    {'answer': 'I\'m Jim Alkhalili and this is The Life Scientific...'}
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This audio question answering pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"audio-question-answering"`.

    The models that this pipeline can use are models that support both audio and text inputs and that have been fine-tuned on
    a variety of tasks, such as Phi-4-multimodal. See the up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?pipeline_tag=audio-question-answering).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_model_type(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)

    def _sanitize_parameters(self, top_k=None, max_new_tokens=None, generate_kwargs=None, **kwargs):
        preprocess_params = {}
        forward_params = {}
        postprocess_params = {}

        if max_new_tokens is not None:
            forward_params["max_new_tokens"] = max_new_tokens
            warnings.warn(
                "`max_new_tokens` is deprecated and will be removed in version 4.49 of Transformers. To remove this warning, pass `max_new_tokens` as a key inside `generate_kwargs` instead.",
                FutureWarning,
            )
        if generate_kwargs is not None:
            if max_new_tokens is not None and "max_new_tokens" in generate_kwargs:
                raise ValueError(
                    "`max_new_tokens` is defined both as an argument and inside `generate_kwargs` argument, please use"
                    " only 1 version"
                )
            forward_params.update(generate_kwargs)
        if top_k is not None:
            postprocess_params["top_k"] = top_k

        return preprocess_params, forward_params, postprocess_params

    def __call__(
        self,
        inputs=None,
        question: str = None,
        audio: Union[str, np.ndarray, bytes, tuple] = None,
        **kwargs
    ):
        """
        Answer a question about audio. The audio can be either a remote URL, a local path, or the audio data itself.

        Args:
            inputs (`Dict[str, Any]`, *optional*):
                A dictionary containing both the question and audio. If provided, overrides question and audio arguments.
            question (`str`, *optional*):
                The question to answer about the audio.
            audio (`str` or `np.ndarray` or `bytes` or `tuple`, *optional*):
                The audio to analyze. It can be:
                - A string containing a http link pointing to an audio file
                - A string containing a local path to an audio file
                - The bytes of an audio file
                - The raw audio data as a numpy array of shape (length,) or (2, length)
                - A tuple containing the raw audio data as numpy array and the sampling rate as int
            max_new_tokens (`int`, *optional*):
                The maximum number of tokens to generate. This can be useful to control the length of the generated answer.
            generate_kwargs (`dict`, *optional*):
                Additional kwargs to pass to the generate method of the model.
            top_k (`int`, *optional*):
                The number of top answers to return. By default, only the top-1 answer is returned.

        Return:
            A dictionary with the following keys:
            - **answer** (`str`) -- The generated answer.
            
            If top_k is set to a value greater than 1, the output will be a list of dictionaries with the above structure.
        """
        # Handle inputs in various formats
        if inputs is None and question is not None and audio is not None:
            inputs = {"question": question, "audio": audio}
        elif isinstance(inputs, dict) and inputs.get("question") is not None and inputs.get("audio") is not None:
            pass  # Use inputs as is
        elif question is not None and audio is not None:
            inputs = {"question": question, "audio": audio}
        else:
            raise ValueError(
                "You must provide either 'inputs' as a dict containing 'question' and 'audio' keys, "
                "or provide both 'question' and 'audio' arguments directly."
            )
            
        return super().__call__(inputs, **kwargs)

    def preprocess(self, inputs):
        # Extract question and audio from inputs
        question = inputs.get("question")
        audio = inputs.get("audio")
        
        if question is None or audio is None:
            raise ValueError("Both 'question' and 'audio' must be provided in the inputs dictionary")
            
        # Process audio input
        if isinstance(audio, str):
            if audio.startswith("http://") or audio.startswith("https://"):
                # Download from URL
                audio_data = urlopen(audio).read()
                audio_data, sampling_rate = sf.read(io.BytesIO(audio_data))
            else:
                # Local file path
                audio_data, sampling_rate = sf.read(audio)
        elif isinstance(audio, bytes):
            # Bytes data
            audio_data, sampling_rate = sf.read(io.BytesIO(audio))
        elif isinstance(audio, tuple) and len(audio) == 2:
            # Tuple of (audio_data, sampling_rate)
            audio_data, sampling_rate = audio
        elif isinstance(audio, np.ndarray):
            # Raw audio data, assume default sampling rate of 16000
            audio_data = audio
            sampling_rate = 16000
        else:
            raise ValueError(
                f"Audio must be a string (URL or local path), bytes, numpy array, or tuple of (audio_data, sampling_rate). Got {type(audio)}"
            )

        # Format the user prompt
        user_prompt = '<|user|>'
        assistant_prompt = '<|assistant|>'
        prompt_suffix = '<|end|>'
        prompt = f'{user_prompt}<|audio_1|>{question}{prompt_suffix}{assistant_prompt}'

        # Process with the model
        model_inputs = self.processor(text=prompt, audios=[(audio_data, sampling_rate)], return_tensors='pt')
        
        if self.torch_dtype:
            model_inputs = model_inputs.to(self.torch_dtype)
        if self.device.type == "cuda":
            model_inputs = model_inputs.to(self.device)

        return model_inputs

    def _forward(self, model_inputs, **generate_kwargs):
        # User-defined `generation_config` passed to the pipeline call takes precedence
        if "generation_config" not in generate_kwargs:
            generate_kwargs["generation_config"] = self.generation_config

        # Generate the answer
        output_ids = self.model.generate(
            **model_inputs,
            **generate_kwargs,
        )
        
        # The output_ids contain the full sequence including the prompt, so we extract only the generated part
        output_ids = output_ids[:, model_inputs["input_ids"].shape[1]:]
        
        return {"output_ids": output_ids}

    def postprocess(self, model_outputs, top_k=1):
        output_ids = model_outputs["output_ids"]
        
        # Decode the output tokens
        decoded_output = self.processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        if top_k == 1:
            return {"answer": decoded_output}
        else:
            # For multiple answers, this would need to be implemented based on model's capability
            # Currently, just return the single answer but wrapped in a list
            return [{"answer": decoded_output}]