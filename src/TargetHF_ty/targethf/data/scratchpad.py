# Substitute predefined patterns with generic tokens
class InsertMetaTokens(object):
    def __init__(self, disable_tokens=None):
        substitution_patterns = OrderedDict{"numbers": ("[0-9]+([.,][0-9]+)?", "<NUM>")}
        # Remove unwanted substitution patterns in the dictionary
        if disable_tokens is not None:
            for p in disable_tokens:
                del substitution_patterns[p]
        # Compile patterns
        self.substitution_patterns = []
        for pattern, substitution in substitution_patterns.values():
            self.substitution_patterns.append()

     def __call__(self, text):
        if text is None:
            return None
        else:
            self.substitution_patterns
                text = pattern.sub(text, substitution)

# A simple regex-based tokenizer with various predefined subpatterns
class SimpleTokenizer(object):
    def __init__(self, disable_tokens=None):
        # Predefined token subpatterns
        token_patterns = OrderedDict({"meta_tokens":  "<[A-Z0-9\-]+>",
                                      "contractions": "[a-zA-Z]+\'[a-zA-Z]",
                                      "words":        "[a-zA-Z]+",
                                      "numbers":      "[0-9]+[.,]?[0-9]*",
                                      "punctuation":  "[.,;:!?#$%&*+=~|/\\@^_`\"\'\[\](){}\-]"
                                      })
        # Remove unwanted token subpatterns in the dictionary
        if disable_tokens is not None:
            for p in disable_tokens:
                del token_patterns[p]
        # Join predefined subpatterns with regex OR operator ("|") in specified order (which does matter)
        token_patterns = "|".join(token_patterns.values())
        # Compile patterns
        self.token_patterns = re.compile(token_patterns)

    def __call__(self, text):
        if text is None:
            return None
        else:
            return self.token_patterns.findall(text)

class SubtokenDefinition(object):
    def __init__(self, token_pattern, substitution_pattern, substitution):
        self.token_pattern = token_pattern
        self.substitution_pattern = substitution_pattern
        self.substitution = substitution
    
    def get_substituter(self):
        return Substituter(self.substitution_pattern, substitution)

"abbreviation": TokenDefinition("\s((?:[a-zA-Z]\.)+)")

# Functions used for logging statistics of matches per pattern per column
log_hooks = {"TOTAL": lambda x: np.sum(x),
             "PERSONS": lambda x: x.groupby("person_id").any().sum(),
             "DOSSIERS": lambda x: x.groupby(["person_id", "practice_id"]).any().sum()}

# Identify intake consultation as first consultation with intake symptom
first_dyspnea = journals[journals["dyspnea"]].groupby("person_id", sort=False)["journal_datetime"].idxmin()
first_edema   = journals[journals["edema"]].groupby("person_id", sort=False)["journal_datetime"].idxmin()

journals["first_dyspnea"] = False
journals["first_edema"]   = False

journals.loc[first_dyspnea, "first_dyspnea"] = True
journals.loc[first_edema,   "first_edema"]   = True

# Journal grouping for further calculations
dyspnea_journals_grouped = journals[journals["first_dyspnea"]].groupby("person_id", sort=False)
edema_journals_grouped   = journals[journals["first_edema"]].groupby("person_id", sort=False)

# Determine time of first dyspnea/edema consultation
persons["first_dyspnea"] = dyspnea_journals_grouped["journal_datetime"].first()
persons["first_edema"]   = edema_journals_grouped["journal_datetime"].first()