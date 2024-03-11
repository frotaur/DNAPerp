import os,shutil
import requests
from tqdm import tqdm
import gzip


def download_from_link(link, out_path):
    if(not os.path.exists(out_path)):

        with requests.get(link, stream=True) as response:
            # Check if the request was successful
            response.raise_for_status()

            total_size_in_bytes = int(response.headers.get('content-length', 0))
            # Open the file in binary write mode
            with open(out_path, 'wb') as file:
                # Download the file in chunks
                for chunk in tqdm(response.iter_content(chunk_size=8192),total=total_size_in_bytes/8192, unit='KB', unit_scale=True ): 
                    file.write(chunk)
    else :
        print('File already exists, skipping download')


def get_links_with_name(file_path):
    with open(file_path, 'r') as file:
        lines = [phrase.strip() for phrase in file.readlines() if phrase.strip() != '']
    
    linkpath = [(split[0], split[1]) for split in [line.split(' ') for line in lines]]

    print('I read : ', linkpath)
    return linkpath


def download_from_file(links_file_path, out_dir='/home/vassilis/bigdrive/vassilis/DNA_data'):
    linkpath = get_links_with_name(links_file_path)
    for link, filename in linkpath:
        download_from_link(link, os.path.join(out_dir,filename))


def extract_gz(folder_path,out_dir=None):
    """
        Extracts all the .gz files in a folder to .fa files

        Args:
            folder_path (str): The folder to search for .gz files
            out_dir (str): The folder to write the .fa files. If None, writes to the same folder as the .gz files
    """
    if(out_dir is None):
        out_dir=''
        
    os.makedirs(os.path.join(folder_path,out_dir),exist_ok=True)

    for filename in os.listdir(folder_path):
        # Construct the full path to your file
        file_path = os.path.join(folder_path, filename)
        
        # Check if the file is a gzip file by attempting to open it
        try:
            with gzip.open(file_path, 'rt') as f_in:
                # Construct the output filename with .fa extension
                output_file_path = os.path.join(folder_path, os.path.join(out_dir,filename + '.fa'))
                
                # Write the decompressed content to a new file with .fa extension
                with open(output_file_path, 'w') as f_out:
                    f_out.write(f_in.read())
    
            print(f"Extracted {filename} to {filename}.fa")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
