"""Local byte-level tokenizer for integration tests without model downloads."""

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast


def make_tokenizer(directory):
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocabulary = {char: index for index, char in enumerate(alphabet)}
    tokenizer = Tokenizer(models.BPE(vocabulary, []))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    wrapper = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    wrapper.save_pretrained(directory)
    return wrapper
