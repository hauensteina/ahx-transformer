class Tokenizer:
    def __init__(self, samples) -> None:
        text = '0' + ''.join(samples)
        chars = sorted(list(set(text)))
        self.load_chars(chars)

    @classmethod
    def from_dict(cls, ddict):
        tokenizer = cls([])
        tokenizer.load_state_dict(ddict)
        return tokenizer    

    def load_chars(self,chars):
        self.vocab_sz = len(chars)
        print(f'''alphabet:{''.join(chars)}''')
        print(f'''alphabet size:{self.vocab_sz}''')
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def ddict(self):
        return {'chars': self.chars }

    def load_dict(self, ddict):
        chars = ddict['chars']
        self.load_chars(chars)

    def encode(self, x):    
        """ Encode one string into a list of tokens """
        return [self.stoi[ch] for ch in x]
    
    def decode(self, x):
        """ Decode a list of tokens """
        return ''.join([self.itos[i] for i in x])
    