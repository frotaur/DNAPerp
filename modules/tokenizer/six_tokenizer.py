import torch
from pathlib import Path
from itertools import product
from .simple_tokenizer import SimpleTokenizer
curpath = Path(__file__).parent.absolute().as_posix()


class SixerTokenizer(SimpleTokenizer):
    """
        Tokenizer which assigns a token for each possible 6-mer of 'a','c','g','t' and 'n'. 
        Exception is '\n', which is a single token.

        Args:
            tok_dict_loc (str): Location of the tokenizer dictionary. Should be a torch.save'd dictionary.
    """
    def __init__(self):
        super().__init__()
        self.make_dict()

        print('Generated 6-mer dictionary, length : ', len(self.tok_dict))


    def make_dict(self):
        """
            Makes a dictionary of all possible 6-mers of 'a','c','g','t' and '\n'
        """
        self.tok_dict = {}
        self.tok_dict['$'] = 0
        self.tok_dict['\n'] = 1
        # Create all possible 6-mers

        for kmer in product('acgtn',repeat=6):
            self.tok_dict[''.join(kmer)] = len(self.tok_dict)

        self.tok_dict_rev = {v:k for k,v in self.tok_dict.items()}
        self.vocab_size = len(self.tok_dict)

    def tokenize(self, text:str, assert_ok=True):
        """
            Tokenizes a string into a torch tensor

            Args:
                text (str): The string to tokenize
                assert_ok (bool): Whether to assert that the provided text is tokenizable
        """
        new_phrase = []

        sentences = text.split('\n') # Split by newlines

        if(assert_ok) :
            for sentence in sentences:
                assert len(sentence) % 6 == 0, f'Length of sentence (except \\n) {sentence} is not a multiple of 6'
                assert set(sentence).issubset(set('acgt')), "String contains characters other than a, c, g, t or \\n"

        try:
            for sentence in sentences:
                for i in range(0,len(sentence),6):
                    new_phrase.append(self.tok_dict[sentence[i:i+6]])
                new_phrase.append(self.tok_dict['\n'])
            new_phrase.pop() # Remove last newline
        except KeyError as e:
            print(f"KeyError: {e}, failed to tokenize text.")
            print("Run with assert_ok=True to make sure text is tokenizable.")
            print(f"Offending sentence: {sentence}")
            print(f"Offending 6-mer: {sentence[i:i+6]}")

            raise e

        return torch.tensor(new_phrase,dtype=torch.int)
    
    def detokenize(self, tensor:torch.Tensor):
        """
            Detokenizes a torch tensor into a string
        """
        return ''.join([self.tok_dict_rev.get(int(i),'<UNK>') for i in tensor])
    

if __name__ == '__main__':
    st = SixerTokenizer()

    tokiboi = st.tokenize('gacgttgcacagtacagatcagac\naaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
    print(tokiboi)
    print(st.detokenize(tokiboi))