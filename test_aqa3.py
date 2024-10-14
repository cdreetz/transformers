import os
import transformers

print(f"Using transformers from: {os.path.dirname(transformers.__file__)}")

from transformers import pipeline


audio_file = r"/Users/christian/dev/tools/first_30_seconds.mp3"
question = "How is the speaker doing?"

#aqa = pipeline("audio-question-answering")
aqa = pipeline("audio-question-answering")
result = aqa(audio_file, question)

print()
print("I think we're getting somewhere")
print()
