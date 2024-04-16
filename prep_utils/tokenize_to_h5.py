"""
    Tokenizes all text files in folder to .h5 files, using a premade dictionary.
"""
from pathlib import Path
import os, sys

projectpath = Path(__file__).parent.parent.as_posix()
sys.path.append(projectpath)
from modules.tok_utils import tok_text_to_h5, merge_text
from modules.tokenizer import SixerTokenizer

if __name__=='__main__':
    num_dict = {}
    texts_folder = 'dataloc'
    tokenizername = 'tokisixer'

    # premade_dict = {"$" : 0, "a" : 1, "c": 2, "g":3, "t":4, "n":5, "\n" : 6} # Dictionary to use as tokenizer.
    
    destination_folder = os.path.join(projectpath,'dataloc','h5data') # Folder to save the .h5 file
    temp_merged = os.path.join(projectpath,'dataloc','human.txt')

    # if(not os.path.exists(temp_merged)):
    #     merge_text(texts_folder, os.path.join(DATA_FOLDER,'..','merged_R.txt'))
    
    toki = SixerTokenizer()

    text = temp_merged
    out_name = 'human_six'
    tok_text_to_h5(text,tokenizer=toki,tokenizer_folder='toki_useless',
                    tokenizer_name=tokenizername, output_name=out_name, 
                    destination_folder=destination_folder, delete_txt=False,
                    dtype='int16')

    # os.remove(temp_merged)