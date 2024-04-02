"""
    Tokenizes all text files in folder to .h5 files, using a premade dictionary.
"""
from modules.tok_utils import tok_text_to_h5, merge_text
from pathlib import Path
import os
from modules.util.data_folder import DATA_FOLDER
curpath = Path(__file__).parent.absolute().as_posix()


if __name__=='__main__':
    num_dict = {}
    texts_folder = 'dataloc'
    tokenizername = 'ordered_dna_char_R'

    premade_dict = {"$" : 0, "a" : 1, "c": 2, "g":3, "t":4, "n":5, "\n" : 6} # Dictionary to use as tokenizer.
    
    destination_folder = os.path.join('dataloc','h5data') # Folder to save the .h5 file
    temp_merged = os.path.join('all_dna_lower_R.txt')

    # if(not os.path.exists(temp_merged)):
    #     merge_text(texts_folder, os.path.join(DATA_FOLDER,'..','merged_R.txt'))
    

    text = temp_merged
    out_name = 'char_ordered_dna_R.h5'
    tok_text_to_h5(text,premade_dict=premade_dict,tokenizer_folder='new_toki_char',
                    tokenizer_name=tokenizername, output_name=out_name, 
                    destination_folder=destination_folder)

    os.remove(temp_merged)