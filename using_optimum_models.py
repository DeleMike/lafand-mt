"""# Prepare models"""

from transformers import AutoProcessor
from transformers import AutoTokenizer
from transformers import T5ForConditionalGeneration
from onnxruntime import InferenceSession
import numpy as np


model_name = "Davlan/oyo-t5-small"

# onnx_enc_path = "./encoder_quant.onnx"
# onnx_dec_path = "./decoder_quant.onnx"

onnx_enc_path = "oyto_t5_small_onnx/encoder_model.onnx"
onnx_dec_path = "oyto_t5_small_onnx/decoder_model.onnx"

# Load feature_extractor and tokenizer
processor = AutoTokenizer.from_pretrained(model_name_or_path)

feature_extractor = processor.feature_extractor
tokenizer = processor.tokenizer

# Start onnxruntime sessions
enc_session = InferenceSession(onnx_enc_path)
dec_session = InferenceSession(onnx_dec_path)

# Preprocess input data
input_text = 'mo fe jeun'


input_features = np.array(feature_extractor(audio, sampling_rate=16000).input_features, dtype=np.float32)
encoder_hidden_states = enc_session.run(None, input_feed={"input_features":input_features})[0]


"""# Decoder"""

input_ids = np.array(processor.get_decoder_prompt_ids(), dtype=np.int64)

input_feed = {"input_ids":input_ids, "encoder_hidden_states":encoder_hidden_states}

out = dec_session.run(None, input_feed=input_feed)

logits = np.array(out[0])

result = []
for item in np.argmax(logits, axis=-1).flatten():
  result.append(item)
print(result)

tokenizer.decode(result)

shapes = [np.array(output).shape for output in out]
output_names = [output.name for output in dec_session.get_outputs()]
outputs = list(zip(output_names, shapes))
outputs



config = T5ForConditionalGeneration.from_pretrained(model_name).config

encoder_layers = config.encoder_layers
encoder_heads = config.encoder_attention_heads

decoder_layers = config.decoder_layers
decoder_heads = config.decoder_attention_heads

d_model = config.d_model

encoder_dim = [1, encoder_heads, 0, int(d_model / encoder_heads)]
decoder_dim = [1, encoder_heads, 0, int(d_model / encoder_heads)]

# encoder_dim = [1, encoder_heads, 1500, int(d_model / encoder_heads)]
# decoder_dim = [1, encoder_heads, 2, int(d_model / encoder_heads)]

print(f"Encoder layers = {encoder_layers}, Decoder layers = {decoder_layers}")
print(encoder_dim)
print(decoder_dim)

"""# Decoder with past model"""

onnx_past_dec_path = "decoder_with_past_model_quantized.onnx"

past_dec_session = InferenceSession(onnx_past_dec_path)

encoder_keys = {}
def step_1_decode_past(decoder_out):
  input_feed = {output_names[i].replace("present", "past_key_values"):out_last_decoder[i] for i in range(len(output_names))}
  input_feed["input_ids"] = np.array([[result[-1]]], dtype=np.int64)
  del input_feed["logits"]

  present_out = past_dec_session.run(None, input_feed=input_feed)
  return present_out

decoder_output_names = list(filter(lambda a : "decoder" in a or "logits" in a, output_names))
print(len(decoder_output_names))
print(decoder_output_names)

def decode_step(out_last_decoder):
  input_feed = {}
  j = 0
  for i in range(len(out)):
    if "encoder" in output_names[i]:
      input_feed[output_names[i]] = out[i]
    else:
       input_feed[output_names[i]] = out_last_decoder[j]
       j += 1

  input_feed["input_ids"] = np.array([[result[-1]]], dtype=np.int64)
  del input_feed["logits"]


def append_token(logits_present):
  for item in np.argmax(logits_present, axis=-1).flatten().flatten():
    result.append(item)

step_2 = decode_step(out)
new_logits = step_2[0]
append_token(new_logits)
result

decode_step(step_2)

len(step_2)

x = [item for sublist in result for item in sublist]
x

tokenizer.decode(x)

input_feed = {output_names[i].replace("present", "past_key_values"):out_last_decoder[i] for i in range(len(output_names))}



"""# Merged decoder"""

decoder_feeds = {"input_ids": input_ids, "encoder_hidden_states":encoder_hidden_states, "use_cache_branch": np.array([True])}


for i in range(encoder_layers):
  decoder_feeds[f"past_key_values.{i}.encoder.key"] = np.empty(encoder_dim, dtype=np.float32)
  decoder_feeds[f"past_key_values.{i}.encoder.value"] = np.empty(encoder_dim, dtype=np.float32)

for i in range(decoder_layers):
  decoder_feeds[f"past_key_values.{i}.decoder.key"] = np.empty(decoder_dim, dtype=np.float32)
  decoder_feeds[f"past_key_values.{i}.decoder.value"] = np.empty(decoder_dim, dtype=np.float32)


merged_decoder_path = "./decoder_merged_quant.onnx"

merged_decoder_session = InferenceSession(merged_decoder_path)
merged_out = merged_decoder_session.run(None, input_feed=decoder_feeds)

logits_merged = np.array(merged_out[0])
np.argmax(logits_merged, axis=-1).flatten()