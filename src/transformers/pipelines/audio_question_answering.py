# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import os
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import requests

from ..models.auto.modeling_auto import MODEL_FOR_AUDIO_QUESTION_ANSWERING_MAPPING_NAMES
from ..utils import add_end_docstrings, is_torch_available, is_tf_available, is_torchaudio_available, logging
from .audio_utils import ffmpeg_read, ffmpeg_microphone
from .base import ChunkPipeline, build_pipeline_init_args


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


@add_end_docstrings(build_pipeline_init_args(has_tokenizer=True, has_feature_extractor=True))
class AudioQuestionAnsweringPipeline(ChunkPipeline):
    """
    Audio Question Answering pipeline using any model that takes audio input and a text question and outputs a text answer.
    This pipeline supports models like Qwen/Qwen2-Audio-7B-Instruct, which take both audio and a text prompt to produce a response.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> audio_qa = pipeline(model="Qwen/Qwen2-Audio-7B-Instruct")
    >>> audio_qa(
    ...     audio="https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
    ...     question="What is being discussed in this audio?",
    ... )
    {'answer': 'The audio discusses dinner preparations, specifically mentioning stew with turnips, carrots, potatoes, and mutton in a peppered flour sauce.'}
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This Audio Question Answering pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"audio-question-answering"`.

    The models that this pipeline can use are models that have been fine-tuned on audio-text tasks.
    See the up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=audio-question-answering).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if self.framework != "pt":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")
        
        self.check_model_type(MODEL_FOR_AUDIO_QUESTION_ANSWERING_MAPPING_NAMES)
        
        if not hasattr(self.model, "generate"):
            raise ValueError(
                f"Model {self.model.__class__.__name__} doesn't have a 'generate' method, necessary for Audio Question Answering."
            )
            
        if self.feature_extractor is None and self.processor is None:
            raise ValueError(
                "Audio Question Answering requires either a feature extractor or a processor, but none was provided."
            )

    def __call__(
        self,
        audio: Union[np.ndarray, bytes, str, Dict],
        question: str = None,
        chunk_length_s: Optional[float] = None,
        stride_length_s: Optional[Union[float, Tuple[float, float]]] = None,
        batch_size: int = 1,
        return_timestamps: Optional[bool] = None,
        generate_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Answer questions about an audio input.

        Args:
            audio (`np.ndarray` or `bytes` or `str` or `dict`):
                The audio input. It can be either:
                - A string containing either a local path to an audio file, or a URL pointing to an audio file.
                - The audio file loaded as `bytes`.
                - A numpy array containing the audio waveform in the expected sample rate.
                - A dictionary with "array" or "raw" field containing the audio waveform and a "sampling_rate" field.
            question (`str`):
                The question to ask about the audio content.
            chunk_length_s (`float`, *optional*):
                The length in seconds of each chunk used for processing long audio files. If provided, the audio
                will be processed in chunks of this length, with some overlap as specified by `stride_length_s`.
                This is useful for processing long audios that would otherwise exceed the model's context length.
            stride_length_s (`float` or `tuple(float, float)`, *optional*):
                The length in seconds of the overlapping stride between consecutive chunks. Can be a tuple
                with (left_stride_length_s, right_stride_length_s). Defaults to `chunk_length_s / 6`.
            batch_size (`int`, *optional*, defaults to 1):
                The number of chunks to process at once when chunking is enabled.
            return_timestamps (`bool`, *optional*):
                Whether to return timestamps with the answer. This is only supported by some models.
            generate_kwargs (`dict`, *optional*):
                Additional keyword arguments to pass to the generation method of the model.

        Return:
            A `dict` containing:
            - **answer** (`str`) -- The answer to the question based on the audio content.
            - **timestamps** (*optional*, depends on model and `return_timestamps`) -- Timestamps associated with the answer.
        """
        if generate_kwargs is None:
            generate_kwargs = {}
        
        self._batch_size = batch_size
        
        # Prepare inputs
        if isinstance(audio, dict) and question is None:
            inputs = audio
            if "question" not in inputs:
                raise ValueError("The audio dictionary must contain a 'question' field.")
            if "audio" not in inputs and not any(k in inputs for k in ["array", "raw", "path", "sampled_audio"]):
                raise ValueError("The audio dictionary must contain an 'audio' field or equivalent.")
        else:
            if question is None:
                raise ValueError("The question parameter cannot be None.")
            if audio is None:
                raise ValueError("The audio parameter cannot be None.")
            inputs = {"audio": audio, "question": question}
        
        # Add chunking parameters
        if chunk_length_s is not None:
            inputs["chunk_length_s"] = chunk_length_s
        if stride_length_s is not None:
            inputs["stride_length_s"] = stride_length_s
        if return_timestamps is not None:
            inputs["return_timestamps"] = return_timestamps
        inputs["generate_kwargs"] = generate_kwargs
        
        return super().__call__(inputs, **kwargs)

    def _sanitize_parameters(
        self,
        chunk_length_s=None,
        stride_length_s=None,
        return_timestamps=None,
        generate_kwargs=None,
        sampling_rate=None,
        timeout=None,
        **kwargs
    ):
        preprocess_params = {}
        
        # Handle chunking parameters
        if chunk_length_s is not None:
            preprocess_params["chunk_length_s"] = chunk_length_s
        if stride_length_s is not None:
            preprocess_params["stride_length_s"] = stride_length_s
        if sampling_rate is not None:
            preprocess_params["sampling_rate"] = sampling_rate
        if timeout is not None:
            preprocess_params["timeout"] = timeout
            
        # Forward parameters (mostly for generation)
        forward_params = {}
        if generate_kwargs is not None:
            forward_params["generate_kwargs"] = generate_kwargs
        
        # Postprocessing parameters
        postprocess_params = {}
        if return_timestamps is not None:
            postprocess_params["return_timestamps"] = return_timestamps
            forward_params["return_timestamps"] = return_timestamps
            
        return preprocess_params, forward_params, postprocess_params

    def preprocess(
        self,
        inputs,
        chunk_length_s=None, 
        stride_length_s=None,
        sampling_rate=None,
        timeout=None,
    ):
        # Extract inputs
        if isinstance(inputs, dict):
            question = inputs.get("question", None)
            audio = inputs.get("audio", None)
            
            # Try alternative audio keys if "audio" not present
            if audio is None:
                for key in ["array", "raw", "path", "sampled_audio"]:
                    if key in inputs:
                        audio = inputs[key]
                        break
        else:
            audio = inputs
            question = None
        
        if question is None or question == "":
            raise ValueError("Question cannot be None or empty.")
            
        # Process the audio input
        if isinstance(audio, str):
            if audio.startswith("http://") or audio.startswith("https://"):
                audio = requests.get(audio, timeout=timeout).content
            else:
                with open(audio, "rb") as f:
                    audio = f.read()
        
        if isinstance(audio, bytes):
            if sampling_rate is None:
                if hasattr(self.feature_extractor, "sampling_rate"):
                    sampling_rate = self.feature_extractor.sampling_rate
                elif hasattr(self.processor, "feature_extractor") and hasattr(self.processor.feature_extractor, "sampling_rate"):
                    sampling_rate = self.processor.feature_extractor.sampling_rate
                else:
                    raise ValueError("Sampling rate must be specified when using raw audio bytes")
            
            audio = ffmpeg_read(audio, sampling_rate)
        
        if isinstance(audio, dict):
            if "sampling_rate" in audio:
                in_sampling_rate = audio["sampling_rate"]
                if "raw" in audio:
                    audio_array = audio["raw"]
                elif "array" in audio:
                    audio_array = audio["array"]
                else:
                    raise ValueError("Audio dict must contain a 'raw' or 'array' field")
                
                if in_sampling_rate != sampling_rate:
                    if is_torchaudio_available():
                        from torchaudio import functional as F
                    else:
                        raise ImportError(
                            "torchaudio is required to resample audio samples. Install it with: `pip install torchaudio`"
                        )
                    audio = F.resample(
                        torch.from_numpy(audio_array), in_sampling_rate, sampling_rate
                    ).numpy()
                else:
                    audio = audio_array
            elif "array" in audio:
                audio = audio["array"]
            elif "raw" in audio:
                audio = audio["raw"]

        if not isinstance(audio, np.ndarray):
            raise ValueError(f"Audio must be a numpy array, bytes, or file path, but got {type(audio)}")
        
        if chunk_length_s is not None:
            if stride_length_s is None:
                stride_length_s = chunk_length_s / 6
            
            chunk_len = int(round(chunk_length_s * sampling_rate))
            stride_left = int(round(stride_length_s * sampling_rate)) if isinstance(stride_length_s, (int, float)) else int(round(stride_length_s[0] * sampling_rate))
            stride_right = int(round(stride_length_s * sampling_rate)) if isinstance(stride_length_s, (int, float)) else int(round(stride_length_s[1] * sampling_rate))
            
            if audio.shape[0] > chunk_len:
                audio_length = audio.shape[0]
                for chunk_start in range(0, audio_length, chunk_len - stride_left - stride_right):
                    chunk_end = min(chunk_start + chunk_len, audio_length)
                    chunk = audio[chunk_start:chunk_end]
                    
                    is_last = chunk_end == audio_length
                    yield self._prepare_model_inputs(chunk, question, sampling_rate, is_last)
            else:
                yield self._prepare_model_inputs(audio, question, sampling_rate, True)
        else:
            yield self._prepare_model_inputs(audio, question, sampling_rate, True)
    
    def _prepare_model_inputs(self, audio, question, sampling_rate, is_last):
        """Helper method to prepare inputs for the model."""
        if self.processor is not None:
            processed = self.processor(
                audio=audio, 
                text=question, 
                sampling_rate=sampling_rate, 
                return_tensors=self.framework
            )
        else:
            audio_features = self.feature_extractor(
                audio, 
                sampling_rate=sampling_rate, 
                return_tensors=self.framework
            )
            
            text_features = self.tokenizer(
                question, 
                return_tensors=self.framework
            )
            
            processed = {**audio_features, **text_features}
        
        if self.framework == "pt" and self.torch_dtype is not None:
            for key, value in processed.items():
                if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                    processed[key] = value.to(dtype=self.torch_dtype)
        
        return {"is_last": is_last, **processed}

    def _forward(self, model_inputs, generate_kwargs=None, return_timestamps=False):
        is_last = model_inputs.pop("is_last", True)
        
        if generate_kwargs is None:
            generate_kwargs = {}
            
        if "generation_config" not in generate_kwargs:
            generate_kwargs["generation_config"] = self.generation_config
        
        if return_timestamps and hasattr(self.model.config, "supports_timestamps") and self.model.config.supports_timestamps:
            generate_kwargs["return_timestamps"] = True
            
        if hasattr(self.model, "supported_generation_modes") and "chat" in self.model.supported_generation_modes:
            if "prompt_template" not in generate_kwargs:
                audio_context = "[AUDIO]"  # This is a placeholder, the actual audio is in the model_inputs
                chat = [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": model_inputs.get("prompt", f"Here's an audio: {audio_context}. Question: {model_inputs.get('text', '')}")}]}]
                generate_kwargs["prompt_template"] = chat
            
            generated_sequence = self.model.generate(**model_inputs, **generate_kwargs)
        else:
            generated_sequence = self.model.generate(**model_inputs, **generate_kwargs)
        
        return {"generated_sequence": generated_sequence, "is_last": is_last}

    def postprocess(self, model_outputs, return_timestamps=False):
        """
        Post-process the model outputs into a user-friendly format.
        
        Args:
            model_outputs: Raw outputs from the model
            return_timestamps: Whether to include timestamps in the output
            
        Returns:
            dict: Processed results with text answer and optional timestamps
        """
        if not isinstance(model_outputs, list):
            model_outputs = [model_outputs]
            
        answers = []
        timestamps = []
        
        for output in model_outputs:
            generated_sequence = output["generated_sequence"]
            
            if hasattr(generated_sequence, "sequences"):
                sequences = generated_sequence.sequences
            else:
                sequences = generated_sequence
                
            if return_timestamps and hasattr(generated_sequence, "timestamps"):
                timestamps.extend(generated_sequence.timestamps)
                
            current_answer = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]
            answers.append(current_answer)
                
        if len(answers) > 1:
            answer = " ".join(answers)
        else:
            answer = answers[0]
            
        result = {"answer": answer}
        
        if return_timestamps and timestamps:
            result["timestamps"] = timestamps
            
        return result
        