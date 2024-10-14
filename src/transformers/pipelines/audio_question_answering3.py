# from transformers import Pipeline, AutoModelForSpeechSeq2Seq, AutoModelForQuestionAnswering
import os
import numpy as np
import torch

from .base import ChunkPipeline
from ..models.auto.modeling_auto import AutoModelForSpeechSeq2Seq, AutoModelForQuestionAnswering, AutoModelForCTC
from ..models.auto.processing_auto import AutoProcessor
from ..models.whisper.processing_whisper import WhisperProcessor
from ..models.whisper.modeling_whisper import WhisperForConditionalGeneration
from ..models.auto.tokenization_auto import AutoTokenizer
from .audio_utils import ffmpeg_read

class AudioQuestionAnsweringPipeline3(ChunkPipeline):
    def __init__(self,
                 #asr_model="facebook/wav2vec2-base-960h", 
                 asr_model="openai/whisper-tiny",
                 qa_model="distilbert/distilbert-base-cased-distilled-squad",
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        #self.asr_processor = AutoProcessor.from_pretrained(asr_model)
        self.asr_processor = WhisperProcessor.from_pretrained(asr_model)
        #self.asr_model = AutoModelForCTC.from_pretrained(asr_model)
        self.asr_model = WhisperForConditionalGeneration.from_pretrained(asr_model)
        #self.asr_model.config.forced_decoder_ids = None
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model)
        self.qa_tokenizer = AutoTokenizer.from_pretrained(qa_model)
        #self.asr_model = asr_model or AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-base")
        #self.qa_model = qa_model or AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
    

    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, inputs):
        if isinstance(inputs, str):
            if not os.path.isfile(inputs):
                raise ValueError(f"Audio file not found: {inputs}")
            else:
                with open(inputs, "rb") as f:
                    inputs = f.read()
        
        if isinstance(inputs, bytes):
            inputs = ffmpeg_read(inputs, self.asr_processor.feature_extractor.sampling_rate)

        if isinstance(inputs, np.ndarray):
            audio_array = inputs
            sampling_rate = self.asr_processor.feature_extractor.sampling_rate

        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")

        processed = self.asr_processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )
        print(f"Processed input features shape: {processed.input_features.shape}")

        return processed

    def _forward(self, model_inputs, question):
        #with torch.no_grad():
        #    asr_logits = self.asr_model(**model_inputs).logits

        #predicted_ids = torch.argmax(asr_logits, dim=-1)
        #transcription = self.asr_processor.batch_decode(predicted_ids)[0]
        print(f"Model inputs shape: {model_inputs.shape}")
        predicted_ids = self.asr_model.generate(model_inputs)
        print(f"Predicted ids shape: {predicted_ids.shape}")
        transcription = self.asr_processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )

        print(f"Question being passed to qa_tokenizer: {question}")
        print(f"Transcription being passed to qa_tokenizer: {transcription[0]}")

        inputs = self.qa_tokenizer(
            question,
            transcription[0],
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        print(f"Inputs after tokenization: {inputs}")

        with torch.no_grad():
            outputs = self.qa_model(**inputs)

        return transcription[0], outputs, inputs


    def postprocess(self, model_outputs):
        transcription, qa_outputs, inputs = model_outputs

        # Get the scores for start and end logits
        start_scores = qa_outputs.start_logits[0]
        end_scores = qa_outputs.end_logits[0]

        # Find the tokens with the highest start and end scores
        start_index = np.argmax(start_scores)
        end_index = np.argmax(end_scores)

        if start_index > end_index:
            start_index, end_index = end_index, start_index

        input_ids = inputs["input_ids"][0].tolist()

        tokens = self.qa_tokenizer.convert_ids_to_tokens(input_ids)

        answer_tokens = tokens[start_index: end_index + 1]

        answer = self.qa_tokenizer.convert_tokens_to_string(answer_tokens)

        answer = answer.replace("[CLS]", "").replace("[SEP]", "").strip()

        confidence = (start_scores[start_index] + end_scores[end_index]) / 2

        print(f"Debug - Start Index: {start_index}, End Index: {end_index}")
        print(f"Debug - Answer Tokens: {answer_tokens}")

        return transcription, answer, confidence

    #def __call__(self, inputs, **kwargs):
    def __call__(self, audio_file: str, question: str, **kwargs):
        print(f"Processing audio file: {audio_file}")
        print(f"Answering question: {question}")
        input_features = self.preprocess(audio_file)

        print(f"Input features type: {input_features}")
        #for key, value in input_features.items():
        #    if hasattr(value, 'shape'):
        #        print(f"Shape of {key}: {value.shape}")
        #    else:
        #        print(f"Value of {key}: {value}")

        transcription, qa_outputs, inputs = self._forward(input_features.input_features, question)
        print(f"QA outputs from forward: {qa_outputs}")
        transcription, answer, confidence = self.postprocess((transcription, qa_outputs, inputs))

        print(f"Transcription: {transcription}")
        print(f"Answer: {answer}")

        return {"answer": "This is a placeholder message", "score": 0.0}
