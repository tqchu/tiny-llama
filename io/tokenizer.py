import sentencepiece as spm

class Tok:
    def __init__(self, model_file):
        self.sp = spm.SentencePieceProcessor(model_file=model_file)
        self.bos_id = self.sp.bos_id()   # usually 1
        self.eos_id = self.sp.eos_id()   # usually 2

    def encode(self, text, add_bos=True, add_eos=False):
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids):
        # drop special tokens so we don't print <s> </s>
        ids = [i for i in ids if i not in (self.bos_id, self.eos_id)]
        return self.sp.decode(ids)
