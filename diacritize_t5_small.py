from transformers import AutoTokenizer, pipeline, PretrainedConfig
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import onnxruntime


def diacritize(text=''):
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

    # call translation pipeline
    onnx_translation = pipeline("translation_unyo_to_dcyo", model=model, tokenizer=tokenizer)


    # text_that_work_1 = "Bi a ba lo osuwon iseda, gbogbo gbolohun ede Yoruba ni a le pin si meji ni gbooro bayii"
    # text_that_work_2 = "Nje omo naa ni owo?"

    # text = 'olorun oba o, ese o'
    # text = "Nje omo naa ni owo?" # mo lo si ile
    # text = 'Ijoba orileede Naijiria ni ijoba China lo n koju aarun korona kii se pe won koriira tabi saida si omo Naijiria tabi alawodudu to n gbe Guangzhou ni China'
    # text = 'Ijoba orileede Naijiria ni ijoba China lo n koju aarun korona'
    # text = 'A ki i binu ori ka fi fila de ibadi.'
    result = onnx_translation(text, max_length = 10000)
    # print(f'Result of translation is = {result[0]["translation_text"]}')
    return result[0]["translation_text"]

# text = "Nje omo naa ni owo?" # mo lo si ile
# output = diacritize(text=text)
# print(f'Result of translation is = {output}')

