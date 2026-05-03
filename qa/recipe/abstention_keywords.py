"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from itertools import product


# A non-comprehensive preliminary set of keywords and phrases to match for abstention detection,
# based on prompting GPT4o to generate abstention phrases and manually categorizing them.
# Additionally, the abstention phrases were cross-checked with the SelfAware paper: https://aclanthology.org/2023.findings-acl.551.pdf.
# Existing works don't usually rely to keyword search to detect abstention; instead,
# they either use an LLM judge to determine whether the response contains abstention,
# or modify a multiple-choice QA dataset to include an additional answer "None of the above".


_CANT = ["can't", "cannot", "not able to", "unable to", "not equipped to", "prefer not to", "not authorized to",
         "not enough to", "not possible to", "impossible to", "difficult to", "hard to"]
_RESPOND = ["respond", "reply", "answer", "say", "verify", "help", "determine", "conclude", "draw conclusion", "know"]
_CANT_RESPOND = [' '.join(x) for x in list(product(_CANT, _RESPOND))]


_DONT_HAVE = [
    "don't have", "do not have", "lack", "can't provide", "cannot provide", "don't possess", "do not possess",
    "limited", "no suitable", "no", "not enough", "no known", "no relevant", "doesn't have", "does not have"]
_DATA = [
    "data", "answer", "an answer", "the answer", "a response", "the response", "response",
    "insight", "knowledge", "confidence", "information", "context", "background", "evidence"]
_DONT_HAVE_DATA = [' '.join(x) for x in list(product(_DONT_HAVE, _DATA))]

_NOT_AVAIL = ["not available", "unavailable", "unknown", "insufficient", "not sufficient", "uncertain", "unclear"]
_DATA_NOTAVAIL = [' is '.join(x) for x in list(product(_DATA, _NOT_AVAIL))] + \
                 [' '.join(x) for x in list(product(_DATA, _NOT_AVAIL))] + \
                 [' '.join(x) for x in list(product(_NOT_AVAIL, _DATA))]


_MAYBE = ['may be', 'might be', 'is', 'could be']
_WRONG = ["wrong", "not accurate", "inaccurate", "misleading"]
_MAYBE_WRONG = [' '.join(x) for x in list(product(_MAYBE, _WRONG))]


_OUT_OF = ["outside", "outside of", "out", "out of", "not within", "not in", "beyond"]
_SCOPE = ["scope", "data", "knowledge", "experience", "training", "training data", "dataset", "capability", "capacity", "area of expertise"]
_OUT_OF_SCOPE = [' '.join(x) for x in list(product(_OUT_OF, _SCOPE))] + [' my '.join(x) for x in list(product(_OUT_OF, _SCOPE))]


_UNCERTAINTY = [
    "not sure",
    "unsure",
    "unclear",
    "not certain",
    "not confident",
]

_DONT_KNOW = ["don't know", "do not know", "not aware of", "not heard", "do not understand", "don't understand", "ambiguous"]


ABSTENTION_KEYWORDS = _CANT_RESPOND + _DONT_HAVE_DATA + _DATA_NOTAVAIL + _MAYBE_WRONG + _OUT_OF_SCOPE + _UNCERTAINTY + _DONT_KNOW
