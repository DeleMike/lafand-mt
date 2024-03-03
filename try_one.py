from transformers import AutoTokenizer, pipeline, PretrainedConfig
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import onnxruntime

from pathlib import Path



def _extract_yoruba_sentences(file_path:str):
    """
    Extract only the Yoruba sentences from the jw300 text
    """
    yoruba_sentences = []

    with open(file_path, 'r', encoding='utf-8', newline='') as file:
        content = file.read()
        yoruba_sentences = content.strip().split('\n') # remove whitespaces, then break per line

    return yoruba_sentences

def diacritize(texts=[]):
    # Load encoder model
    encoder_session = onnxruntime.InferenceSession('quantized_oyto_t5_small_onnx/encoder_model_quantized.onnx')

    # Load decoder model
    decoder_session = onnxruntime.InferenceSession('quantized_oyto_t5_small_onnx/decoder_model_quantized.onnx')

    model_id = "quantized_oyto_t5_small_onnx"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    config = PretrainedConfig.from_json_file('quantized_oyto_t5_small_onnx/config.json')
    
     # initialise model
    model = ORTModelForSeq2SeqLM(
        config=config,
        onnx_paths=['quantized_oyto_t5_small_onnx/decoder_model_quantized.onnx','quantized_oyto_t5_small_onnx/encoder_model_quantized.onnx'],
        encoder_session=encoder_session,
        decoder_session=decoder_session, 
        model_save_dir='quantized_oyto_t5_small_onnx',
        use_cache=False, 
    )
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    
    gen_tokens = model.generate(**inputs, use_cache=True, max_new_tokens=100000)
    outputs = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    
    # Write outputs to file
    with open("/mnt/disk/makindele/lafand-mt/diacritse_output.txt", "w", encoding="utf-8") as output_file:
        for text in outputs:
            output_file.write(text + "\n")
    
    print("Outputs written to file: diacritse_output.txt")
    return outputs

   

    # # call translation pipeline
    # onnx_translation = pipeline("translation_unyo_to_dcyo", model=model, tokenizer=tokenizer)

    # result = onnx_translation(text, max_length = 10000)
    # # print(f'Result of translation is = {result[0]["translation_text"]}')
    # return result[0]["translation_text"]

# text = "Nje omo naa ni owo?" # mo lo si ile
# output = diacritize(text=text)
# print(f'Result of translation is = {output}')

absolute_path = Path('/mnt/disk/makindele/data_prep_eng/data_prep_eng/output_data/test_with_no_diacritics.txt').resolve()
diacritize(texts=_extract_yoruba_sentences(file_path=absolute_path)[:])
