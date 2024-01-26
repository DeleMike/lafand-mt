import onnxruntime
import numpy as np

# Load encoder model
encoder_session = onnxruntime.InferenceSession('oyto_t5_small_onnx/encoder_model.onnx')

# Load decoder model
decoder_session = onnxruntime.InferenceSession('oyto_t5_small_onnx/decoder_model.onnx')

# Preprocess input data
input_data = preprocess_input('mo fe jeun')

# Run inference on encoder
encoded_context = encoder_session.run(None, {'input_ids': input_data})

# Run inference on decoder with the encoded context
output_sequence = decoder_session.run(None, {'context_vector': encoded_context})

# Post-process output
final_output = postprocess_output(output_sequence)

print("Final Output:", final_output)