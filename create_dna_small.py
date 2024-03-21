from modules.util import DATA_FOLDER
import random
from tqdm import tqdm

full_data_loc = 'merged.txt'
numchar_full = 36467813576 # (counted with wc -m)
small_data_loc = 'small.txt'

phrase_length = int(1e5) # 100k
target_size = int(5e9) # 1G
num_chunks = numchar_full // phrase_length
cut_length = num_chunks * phrase_length

chunk_list = [i*phrase_length for i in range(num_chunks)] # (starting positions of chunks)

current_size = 0
progress_bar = tqdm(total=target_size, desc="Creating small file", unit="B", unit_scale=True)
with open(full_data_loc,'r') as f:
    with open(small_data_loc,'w') as g:
        # Choose a random chunk
        start = random.choice(chunk_list)
        f.seek(start)
        # Read the chunk
        chunk = f.read(phrase_length)
        g.write(chunk)
        current_size += phrase_length
        progress_bar.update(phrase_length)