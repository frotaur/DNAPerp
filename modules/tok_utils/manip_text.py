import os
from tqdm import tqdm

def merge_text(text_folder, output_file_path, chunk_size=1024*1024):
    """
    Merges multiple large files into a single file.

    Args :
        text_folder (str) : Folder containing the text files to merge.
        output_file_path (str) : Path to save the merged file.
        chunk_size (int, optional) : Size of the chunks to read and write. Defaults to 1024*1024.
    """

    with open(output_file_path, 'wb') as outfile:
        for file_path in os.listdir(text_folder):
            if(file_path.endswith('.txt')):
                with open(os.path.join(text_folder,file_path), 'rb') as infile:
                    while True:
                        chunk = infile.read(chunk_size)
                        if not chunk:
                            break
                        outfile.write(chunk)

def lower_text(text_loc) :
    """
        Takes a text file, and save a _lower version, where all chars are 'lowered'.
        Will crash if big text with no new_lines.
    """

    with open(os.path.join(os.path.dirname(text_loc),os.path.basename(text_loc)[:-4]+'_lower.txt'),'w') as outfile:
        with open(text_loc,'r') as infile:
            for line in tqdm(infile):
                outfile.write(line.lower())

def k_div_text(text_loc,k, verbose=False):
    """
        Removes characters (except \n) such that each sentence is divisible by k (not counting \n).
        Crashes if file too big and no newlines
    """
    hits =0
    totlines = 0
    with open(os.path.join(os.path.dirname(text_loc),os.path.basename(text_loc)[:-4]+f'_{k}div.txt'),'w') as outfile:
        with open(text_loc,'r') as infile:
            for line in tqdm(infile):
                totlines+=1
                if(len(line)>0 and (len(line)-1)%k!=0):
                    assert line[-1]=='\n', f'Wrong size line, but no \\n ? line : {line}'
                    n = len(line)-1 # Remove the \n
                    n = (n//k)*k # Closest multiple of k

                    outfile.write(line[:n]+'\n')
                    hits+=1
                    if(verbose):
                        print(f'Line : {line}')
                        print(f'Fix : {line[:n]}\\n')
                else :
                    outfile.write(line)
    
    print('Total misbehaved lines : ', hits)
    print(f'So, {hits/totlines*100}% of misbehaved lines')