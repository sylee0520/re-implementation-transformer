import argparse
import sentencepiece as spm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', default=16000, type=int)
    parser.add_argument('--model_type', default='bpe')
    parser.add_argument('--model_prefix', default='spm')
    parser.add_argument('--input_file', default='./iwslt14.tokenized.de-en/tmp/train.en-de')

    args = parser.parse_args()
    
    return args


class Tokenizer():
    def __init__(self) -> None:
        self.templates = '--input={} --model_prefix={} --vocab_size={} --model_type={} --bos_id=2 --eos_id=3 --pad_id=0 --unk_id=1'
        self.spm_path = './sp/'

    def fit(self, input_file, model_prefix, vocab_size, model_type):
        arguments = self.templates.format(input_file, model_prefix, vocab_size, model_type)
        spm.SentencePieceTrainer.Train(arguments)
    
    def load_model(self):
        model_path = self.spm_path + 'spm.model'
        self.spm = spm.SentencePieceProcessor()
        self.spm.Load(model_path)

        return self
    
    def encode(self, sentences, max_length=0):
        encoded_sentences = self.spm.Encode(sentences)
        padded_encoded_sentences = [0] * max_length
        if max_length > 0:
            padded_encoded_sentences[:min(max_length, len(encoded_sentences))] = encoded_sentences[:min(max_length, len(encoded_sentences))]
            encoded_sentences = padded_encoded_sentences

        return encoded_sentences   

    def decode(self, encoded_sentences):
        decoded_sentences = self.spm.Decode(encoded_sentences)

        return decoded_sentences


def main():
    args = get_args()
    tokenizer = Tokenizer()
    print("Tokenizer starts training!")
    tokenizer.fit(args.input_file, args.model_prefix, args.vocab_size, args.model_type)
    print("Tokenizer training is done! ")


if __name__ == "__main__":
    main()

    