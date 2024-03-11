# Runs in a directory, and converts all the .fa files there into .clean.txt files

import os

sub_dict = dict(A='A', C='C', G='G', T='T', a='A', c='C', g='G', t='T')

for file_name in os.listdir('.'):
    if not file_name.endswith('.fa'): continue
    in_file_name = file_name
    out_file_name = f"{in_file_name[:in_file_name.rfind('.')]}.clean.txt"
    n_junk_chars = 0
    n_good_chars = 0
    in_file_size = os.path.getsize(in_file_name)
    i = 0
    print(f"{in_file_name=} {in_file_size=}")
    with open(in_file_name) as in_file, open(out_file_name, 'w') as out_file:
        while True:
            i += 1
            if i % 10000000 == 0: print(f"{100 * i / in_file_size}% done")
            c = in_file.read(1)
            if c in sub_dict: 
                good_char = sub_dict[c]
                out_file.write(good_char)
            else: 
                n_junk_chars += 1
            if not c: break
    out_file_size = os.path.getsize(out_file_name)
    print(f"{out_file_name=} {out_file_size=}")
