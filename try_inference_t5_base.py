from transformers import pipeline
from pathlib import Path

import time 
 
# store starting time 
begin = time.time() 

def _extract_yoruba_sentences(file_path:str):
    """
    Extract only the Yoruba sentences from the jw300 text
    """
    yoruba_sentences = []

    with open(file_path, 'r', encoding='utf-8', newline='') as file:
        content = file.read()
        yoruba_sentences = content.strip().split('\n') # remove whitespaces, then break per line

    return yoruba_sentences

def diacritize(file_no, texts=[]):
    model_name_or_path = "oyot5_base_unyo_dcyo_mix"
    pipe = pipeline("text2text-generation", model=model_name_or_path)
    
    diacritized_content = []
    print(f'About to process {len(texts)} sentences')
    for index, text in enumerate(texts):
        print(f'Processing sentence sentence {index + 1}')
        diacritized_content.append(pipe(text)[0]["generated_text"])
        # print(pipe(text)[0]["generated_text"])
    
    # Write outputs to file
    with open(f"/mnt/disk/makindele/lafand-mt/sample_out_t5_base.txt", "w", encoding="utf-8") as output_file:
        for text in diacritized_content:
            output_file.write(text + "\n")
    
    
    # print(f"Outputs written to file: {name_of_file}")
    # return outputs

absolute_path = Path('/mnt/disk/makindele/data_prep_eng/data_prep_eng/output_data/test_with_no_diacritics.txt').resolve()
# file_num = 640
diacritize(texts=_extract_yoruba_sentences(file_path=absolute_path)[:], file_no=0 )

# store end time 
end = time.time() 
 
# total time taken 
print(f"Total runtime of the program is {end - begin} seconds") 
