import torch
from tqdm import tqdm
import os, shutil
from ..tokenizer.simple_tokenizer import SimpleTokenizer

def tokenize(text_loc, tokenizer : SimpleTokenizer, output_name='last_tokenization', 
             char_level = False,max_lines= 1e6):
    """
        Tokenizes text to .pt files, using the given tokenizer.
        Saves results in tokendata folder, with name 'output_name'.
        WARNING : it will overwrite previous tokenizations, it same name.

        Args:
            text_loc (str): Location of text to tokenize
            tokenizer (SimpleTokenizer): Tokenizer to use  
            output_name (str, optional): Name of the tokenization. Defaults to 'last_tokenization'.
            premade_dict (dict, optional): Dictionary to use as tokenizer. Defaults to None, in which case a new one is created.
            fixed (bool, optional): If True, will raise an error if a character is not in the dictionary. Defaults to False.
            char_level (bool, optional): If True, will tokenize character by character. 
                Otherwise, will tokenize by line. Defaults to False. Only useful if file
                has no/almost no newlines, and is very large.
            max_lines (int, optional): Number of lines to save in each pt file. Defaults to 1e7.
    """
    if(char_level):
        x=50000 # Num of characters to handle at once

    line_tensors = []
    tokendir = os.path.join('tokendata',output_name)
    shutil.rmtree(tokendir,ignore_errors=True)
    os.makedirs(tokendir,exist_ok=True)

    total_size = os.path.getsize(text_loc)

    with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
        with open(text_loc,'r') as file:
            numlines = 0
            if(char_level):
                while True:
                    chunk = file.read(x)
                    if not chunk:
                        break
                    
                    pbar.update(len(chunk.encode('utf-8')))  # Update progress bar with bytes read

                    numlines+=1
                    new_phrase = tokenizer.tokenize(chunk) # (T,) tokenized tensor
                    
                    line_tensors.append(new_phrase)
                    if(numlines%max_lines==0):
                        torch.save(torch.cat(line_tensors,dim=0),os.path.join(tokendir,f'Data{int(numlines//max_lines)}.pt'))
                        line_tensors = []
            else :
                for line in file:
                    pbar.update(len(line.encode('utf-8')))	# Update progress bar with bytes read
                    numlines+=1
                    line_tensors.append(tokenizer.tokenize(line))
                    if(numlines%max_lines==0):
                        torch.save(torch.cat(line_tensors,dim=0),os.path.join(tokendir,f'Data{int(numlines//max_lines)}.pt'))
                        line_tensors = []
                    
    if(len(line_tensors)>0):
        torch.save(torch.cat(line_tensors,dim=0),os.path.join(tokendir,f'Data{int(numlines//max_lines+1)}.pt'))
