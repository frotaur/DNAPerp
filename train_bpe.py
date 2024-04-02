from modules.tokenizer import BPETokenizer


tokenizer = BPETokenizer()
text_location = 'small.txt'
merge_num = 50000

with open(text_location, 'r') as f:
    print('Reading file...')
    text = f.read()
    print('file read')
    tokenizer.train(text, 256 + merge_num) # 256 are the byte tokens, then do 3 merges
    print(tokenizer.encode(text[0:200]))
    # [258, 100, 258, 97, 99]
    print(tokenizer.decode([257,258,259,1024]))
    # aaabdaaabac
    tokenizer.save("DNA_test")
    # writes two files: toy.model (for loading) and toy.vocab (for viewing)