import re
import numpy as np
import pandas as pd
from collections import OrderedDict

# Object for holding subtoken definitions
class TokenDefinition(object):
    def __init__(self, token_pattern, substitution_pattern, substitution, spare_meta_tokens=True):
        # Do not process within <> brackets
        if spare_meta_tokens:
            substitution_pattern = substitution_pattern + "(?!\S*>)"
        self.token_pattern = token_pattern
        self.substitutor = (re.compile(substitution_pattern), substitution)

# A simple regex-based tokenizer with various predefined subpatterns and possible substitutions
class AdditiveTokenizer(object):
    def __init__(self, substitute=[], ignore=[]):
        self.substitutors = []
        self.token_patterns = []

        # Mild preprocessing
        self.substitutors.append((re.compile("[’`]"), "\'"))

        # Predefined token patterns in specified order (which does matter)
        token_defs = OrderedDict(
           {"meta_token":   TokenDefinition("<[a-zA-Z]+(?:\-[0-9]+)?>",     "<([a-zA-Z]+)(\-[0-9]+)?>", r"<\1>"),
            # Substitution concatenates, ignore treats as separate words/tokens
            "contraction":  TokenDefinition("[a-zA-Z]+\'[a-zA-Z]+",         "([a-zA-Z]+)\'([a-zA-Z]+)", r"\1\2"),
            # Substitution concatenates, ignore treats as separate words/tokens
            "hyphenation":  TokenDefinition("[a-zA-Z]+(?:\-[a-zA-Z]+)+",    "([a-zA-Z])\-([a-zA-Z])",   r"\1\2"),
            "number":       TokenDefinition("[0-9]*[.,]?[0-9]+",            "[0-9]*[.,]?[0-9]+",        "<NUM>"),
            # Substitution provides basic end-of-sentence detection
            "punctuation":  TokenDefinition("[\.,;:!?#$%&*+=~|/\\@^_\"\'\[\](){}\-]+", "(?<!\.[a-zA-Z])[\.,]+(?![0-9])(?![a-zA-Z]\.)|[;!?]+", "</s>")
            })
        
        # Loop over values and decide between token substitution, token search patterns, or nothing
        for key, value in token_defs.items():
            if key in substitute:
                self.substitutors.append(value.substitutor)
            elif key not in ignore:
                self.token_patterns.append(value.token_pattern)
        
        # Add the basic word token pattern last
        self.token_patterns.append("[a-zA-Z]+")

        # Join token subpatterns with regex OR operator ("|") and compile
        self.token_patterns = re.compile("|".join(self.token_patterns))

    def __call__(self, text):
        if pd.isnull(text):
            return []
        else:
            # Substitute
            for pattern, substitution in self.substitutors:
                text = pattern.sub(substitution, text)
            # Search patterns and return list
            return self.token_patterns.findall(text)

def get_sentence_vector(word_vectors, normalize=True):
    # Divide each word vector by its L2 norm
    if normalize:
        word_vectors = word_vectors/np.expand_dims(np.linalg.norm(word_vectors, ord=2, axis=1), 1)
    # Meanpooling across words
    return np.mean(word_vectors, axis=0)