from tqdm import tqdm 

def reverse_file(input_file, output_file, buffer_size=4096):
    """
    Reverse a large file line by line in text mode.

    :param input_file: Path to the input file.
    :param output_file: Path to the output file.
    """
    with open(input_file, 'r', encoding='utf-8') as f_in:
        f_in.seek(0, 2)  # Go to the end of the file
        position = f_in.tell()
        current_line = ''
        read_chars =0 
        file_length = position
        last_retreat = min(position,buffer_size)
        position-=min(position,buffer_size)

        with tqdm(total=file_length, desc="Processing file", unit="char", unit_scale=True) as pbar:
            with open(output_file, 'w', encoding='utf-8') as f_out:
                while position >= 0:
                    f_in.seek(position)
                    try:
                        char = f_in.read(last_retreat)
                        read_chars += last_retreat
                        pbar.update(last_retreat)
                    except UnicodeDecodeError:
                        # In case of a split multi-byte character
                        position -= 1
                        pbar.update(1)
                        last_retreat+=1
                        print('ERROR')
                        continue

                    if read_chars > buffer_size:
                        f_out.write((char+current_line)[::-1])
                        read_chars = 0
                        current_line = ''
                    else:
                        current_line = char+ current_line
                    
                    last_retreat = min(buffer_size,position)
                    if(position==0):
                        position = -1
                    position -= last_retreat

                # Write the last line if there is one
                if current_line:
                    f_out.write(current_line[::-1])

# Usage
reverse_file('merged.txt', 'merged_R.txt')
