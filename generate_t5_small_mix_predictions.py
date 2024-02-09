from transformers import AutoTokenizer, pipeline, PretrainedConfig
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import onnxruntime

from pathlib import Path

def diacritize(text=''):
    # Load encoder model
    encoder_session = onnxruntime.InferenceSession('oyto_t5_small_onnx/encoder_model.onnx')

    # Load decoder model
    decoder_session = onnxruntime.InferenceSession('oyto_t5_small_onnx/decoder_model.onnx')

    model_id = "oyto_t5_small_onnx"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    config = PretrainedConfig.from_json_file('oyto_t5_small_onnx/config.json')

    # initialise model
    model = ORTModelForSeq2SeqLM(
        config=config,
        onnx_paths=['oyto_t5_small_onnx/decoder_model.onnx','oyto_t5_small_onnx/encoder_model.onnx'],
        encoder_session=encoder_session, 
        decoder_session=decoder_session, 
        model_save_dir='oyto_t5_small_onnx',
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

def _extract_yoruba_sentences(file_path:str):
    """
    Extract only the Yoruba sentences from the jw300 text
    """
    yoruba_sentences = []

    with open(file_path, 'r', encoding='utf-8', newline='') as file:
        content = file.read()
        yoruba_sentences = content.strip().split('\n') # remove whitespaces, then break per line

    return yoruba_sentences

def run_model_on_undiacritised_file():
    absolute_path = Path('/mnt/disk/makindele/data_prep_eng/data_prep_eng/output_data/test_with_no_diacritics.txt').resolve()
    yoruba_sentences = _extract_yoruba_sentences(absolute_path)
    diacritized_sentences = []
    count = 0
    
    for sentence in yoruba_sentences[:43]:
        print('Currently processing  = ', sentence)
        diacritised_output = diacritize(text=sentence)
        diacritized_sentences.append(diacritised_output)
        count = count+1
        print(f'Count is ={count}' )
        
    # write to another file
    try:
        new_path =  file_path = Path('/mnt/disk/makindele/lafand-mt/t5_small_onnx.txt').resolve()
        with open(new_path, 'w', encoding='utf-8') as new_file:
            new_file.write('\n'.join(diacritized_sentences))
    except:
        print(f'Something happened. We could not write to file')

run_model_on_undiacritised_file()