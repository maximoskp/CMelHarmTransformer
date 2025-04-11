from data_utils import StructGPTMelHarmDataset, PureGenCollator
import os
import numpy as np
from harmony_tokenizers_m21 import ChordSymbolTokenizer, RootTypeTokenizer, \
    PitchClassTokenizer, RootPCTokenizer, GCTRootPCTokenizer, \
    GCTSymbolTokenizer, GCTRootTypeTokenizer, MelodyPitchTokenizer, \
    MergedMelHarmTokenizer
from torch.utils.data import DataLoader
from transformers import AutoConfig, GPT2LMHeadModel,\
                    LogitsProcessor, StoppingCriteria, StoppingCriteriaList
import torch
import torch
from torch.optim import AdamW
from tqdm import tqdm
import argparse
import pickle
import csv

tokenizers = {
    'ChordSymbolTokenizer': ChordSymbolTokenizer,
    'RootTypeTokenizer': RootTypeTokenizer,
    'PitchClassTokenizer': PitchClassTokenizer,
    'RootPCTokenizer': RootPCTokenizer,
    'GCTRootPCTokenizer': GCTRootPCTokenizer,
    'GCTSymbolTokenizer': GCTSymbolTokenizer,
    'GCTRootTypeTokenizer': GCTRootTypeTokenizer
}

def main():

    # Create the argument parser
    parser = argparse.ArgumentParser(description='Script for MLM training a tiny RoBERTa model with a specific harmonic tokenizer.')

    # Define arguments
    parser.add_argument('-t', '--tokenizer', type=str, help='Specify the tokenizer name among: ' + repr(tokenizers.keys()), required=True)
    parser.add_argument('-v', '--dataval', type=str, help='Specify the full path to the root folder of the validation xml/mxl files', required=True)
    parser.add_argument('-g', '--gpu', type=int, help='Specify whether and which GPU will be used by used by index. Not using this argument means use CPU.', required=False)
    parser.add_argument('-s', '--num_beams', type=int, help='Number of beams. Defaults to 5.', required=False)
    parser.add_argument('-p', '--temperature', type=float, help='Temperature, defaults to 0, i.e., no sampling.', required=False)
    parser.add_argument('-b', '--batchsize', type=int, help='Specify batch size. Defaults to 16.', required=False)
    
    # Parse the arguments
    args = parser.parse_args()
    tokenizer_name = args.tokenizer
    # root_dir = '/media/maindisk/maximos/data/hooktheory_xmls'
    val_dir = args.dataval
    device_name = 'cpu'
    if args.gpu is not None:
        if args.gpu > -1:
            device_name = 'cuda:' + str(args.gpu)
    batchsize = 16
    if args.batchsize:
        batchsize = args.batchsize
    num_beams = 5
    if args.num_beams:
        num_beams = args.num_beams
    temperature = 0.0
    if args.temperature:
        temperature = args.temperature
    
    def is_sublist_contiguous(q, d):
        q_len = len(q)
        for i in range(len(d) - q_len + 1):
            if d[i:i + q_len] == q:
                return True
        return False

    melody_tokenizer = MelodyPitchTokenizer.from_pretrained('saved_tokenizers/MelodyPitchTokenizer')
    harmony_tokenizer = tokenizers[tokenizer_name].from_pretrained('saved_tokenizers/' + tokenizer_name)

    tokenizer = MergedMelHarmTokenizer(melody_tokenizer, harmony_tokenizer)
    
    val_dataset = StructGPTMelHarmDataset(val_dir, tokenizer, max_length=512, return_harmonization_labels=True, num_bars=8)
    collator = PureGenCollator(tokenizer)

    valloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, collate_fn=collator)

    model_path = 'saved_models/gpt/' + tokenizer_name + '/' + tokenizer_name + '.pt'

    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer.vocab),
        n_positions=512,
        n_layer=8,
        n_head=8,
        pad_token_id=tokenizer.vocab[tokenizer.pad_token],
        bos_token_id=tokenizer.vocab[tokenizer.bos_token],
        eos_token_id=tokenizer.vocab[tokenizer.eos_token],
        n_embd=512
    )

    model = GPT2LMHeadModel(config)

    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print('Selected device not available: ' + device_name)
    
    checkpoint = torch.load(model_path, map_location=device_name, weights_only=True)
    model.load_state_dict(checkpoint)

    model.eval()
    model.to(device)

    output_folder = 'tokenized/gpt_beam_' + str(num_beams) + '_temp_' + str(temperature).replace('.','x') + '/'

    os.makedirs(output_folder, exist_ok=True)

    tokenized = {
        'melodies': [],
        'real': [],
        'generated': [],
        'constraints': [],
        'success': []
    }
    result_fields = ['melody', 'real', 'generated', 'constraints', 'success']
    with open( output_folder + tokenizer_name + '.csv', 'w' ) as f:
        writer = csv.writer(f)
        writer.writerow( result_fields )
    batch_num = 0
    running_accuracy = 0
    constraint_accuracy = 0
    with torch.no_grad():
        with tqdm(valloader, unit='batch') as tepoch:
            tepoch.set_description(f'run')
            for batch in tepoch:
                for b in batch['input_ids']:
                    melody_tokens = []
                    real_tokens = []
                    generated_tokens = []
                    # find the start harmony token
                    start_harmony_position = np.where( b == tokenizer.vocab[tokenizer.harmony_tokenizer.start_harmony_token] )[0][0]
                    real_ids = b
                    input_ids = b[:(start_harmony_position+1)].to(device)
                    for i in input_ids:
                        melody_tokens.append( tokenizer.ids_to_tokens[ int(i) ].replace(' ','x') )

                    for i in range(start_harmony_position, len(real_ids), 1):
                        if real_ids[i] != tokenizer.pad_token_id:
                            real_tokens.append( tokenizer.ids_to_tokens[ int(real_ids[i]) ].replace(' ','x') )
                    
                    # Define the bar token ID, eos_token_id, and per-batch sequence constraints
                    bar_token_id = tokenizer.vocab['<bar>']
                    eos_token_id = tokenizer.eos_token_id
                    bars_count = (batch['input_ids'] == bar_token_id).sum(dim=1).reshape(batch['input_ids'].shape[0],-1)
                    bars_count = bars_count[0]

                    do_sample = temperature > 0

                    outputs = model.generate(
                        input_ids=input_ids.reshape(1, input_ids.shape[0]),
                        eos_token_id=tokenizer.eos_token_id,
                        max_length=model.config.max_position_embeddings,
                        num_beams=num_beams,
                        do_sample=do_sample,
                        temperature=1 if not do_sample else temperature,
                    )
                    for i in range(start_harmony_position, len(outputs[0]), 1):
                        generated_tokens.append( tokenizer.ids_to_tokens[ int(outputs[0][i]) ].replace(' ','x') )
                    
                    # check whether constraint was achieved
                    # find where melody ends
                    if '</m>' in melody_tokens:
                        melody_end_index = melody_tokens.index('</m>')
                        # keep melody and constraints after end index
                        after_melody = melody_tokens[melody_end_index:]
                        # find how many bars before constraint and keep the constraints
                        bars_count = 0
                        constraint_tokens = []
                        constraint_found = False
                        i = 0
                        while i < len(after_melody):
                            if after_melody[i] == '<bar>':
                                bars_count += 1
                            i += 1
                            while i < len(after_melody) and '<bar>' != after_melody[i] and '<fill>' != after_melody[i]:
                                constraint_found = True
                                constraint_tokens.append(after_melody[i])
                                i += 1
                            if constraint_found:
                                break
                        # get proper bar in generated tokens
                        bar_idxs = [i for i in range(len(generated_tokens)) if generated_tokens[i] == '<bar>']
                        if bars_count - 1 >= len(bar_idxs):
                            start_index = -1 # not applicable
                        else:
                            start_index = bar_idxs[bars_count-1]
                            if bars_count - 1 == len(bar_idxs) - 1:
                                end_index = len(generated_tokens)
                            else:
                                end_index = bar_idxs[bars_count]
                        constraints_area = None
                        if start_index >= 0:
                            constraints_area = generated_tokens[start_index:end_index]
                        res = False if constraints_area is None else is_sublist_contiguous(constraint_tokens, constraints_area)
                        # update constraint accuracy
                        batch_num += 1
                        running_accuracy += res
                        constraint_accuracy = running_accuracy/batch_num
                        tepoch.set_postfix(accuracy=constraint_accuracy)
                        
                        with open( output_folder + tokenizer_name + '.csv', 'a' ) as f:
                            writer = csv.writer(f)
                            writer.writerow( [' '.join(melody_tokens), ' '.join(real_tokens), \
                                    ' '.join(generated_tokens), ' '.join(after_melody), str(res)] )

                        tokenized['melodies'].append( melody_tokens )
                        tokenized['real'].append( real_tokens )
                        tokenized['generated'].append( generated_tokens )
                        tokenized['constraints'].append( after_melody )
                        tokenized['success'].append( res )
    # save all results to csv
    with open(output_folder + tokenizer_name + '.pickle','wb') as handle:
        pickle.dump(tokenized, handle, protocol=pickle.HIGHEST_PROTOCOL)
# end main

if __name__ == '__main__':
    main()