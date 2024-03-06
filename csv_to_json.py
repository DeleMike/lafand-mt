import pandas as pd
import os
import jsonlines

def create_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def export_json_files(output_dir, filename, df, direction='en-sw'):

    to_be_saved = []
    src_data = df['source_lang'].values
    tgt_data = df['target_lang'].values
    src_lang, tgt_lang = direction.split('-')
    N_sent = df.shape[0]
    for s in range(N_sent):
        text_string = {"translation": {src_lang:str(src_data[s]), tgt_lang:str(tgt_data[s])}}
        to_be_saved.append(text_string)

    with jsonlines.open(os.path.join(output_dir, filename), 'w') as writer:
        writer.write_all(to_be_saved)


def combine_texts_lafand(input_path, output_path, direction='en-sw', n_sent=50000):

    df_train = pd.read_csv(input_path + 'new_train.tsv', sep='\t') #[For CSV, use sep=',']
    df_dev = pd.read_csv(input_path + 'dev.tsv', sep='\t')
    df_test = pd.read_csv(input_path + 'test.tsv', sep='\t')

    sc_train, tg_train = df_train.iloc[:n_sent, 1].values, df_train.iloc[:n_sent, 0].values
    sc_dev, tg_dev = df_dev.iloc[:, 1].values, df_dev.iloc[:, 0].values
    # print(f'sc_dev = {sc_dev} and tc_dev = {tg_dev}')
    sc_test, tg_test = df_test.iloc[:, 1].values, df_test.iloc[:, 0].values

    df_train_sctg = pd.DataFrame(sc_train, columns=['source_lang'])
    df_train_sctg['target_lang'] = tg_train

    # dev data
    df_dev_sctg = pd.DataFrame(sc_dev, columns=['source_lang'])
    df_dev_sctg['target_lang'] = tg_dev

    # test data
    df_test_sctg = pd.DataFrame(sc_test, columns=['source_lang'])
    df_test_sctg['target_lang'] = tg_test

    # output data
    output_dir = output_path
    create_dir(output_dir)

    export_json_files(output_dir, 'new_menyo_train.json', df_train_sctg, direction=direction)
    export_json_files(output_dir, 'dev.json', df_dev_sctg, direction=direction)
    export_json_files(output_dir, 'test.json', df_test_sctg, direction=direction)





if __name__ == "__main__":
    input_path = '/mnt/disk/makindele/data_prep_eng/data_prep_eng/output_data/' # [REPLACE with path/directory to the CSV/TSV]
    output_path = '/mnt/disk/makindele/data_prep_eng/data_prep_eng/output_data/' # [REPLACE with output directory to the CSV/TSV]
    combine_texts_lafand(input_path, output_path, direction='unyo-dcyo')  # replace direction with whatever translation direction