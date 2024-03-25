from cleantext import clean
import re
from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
inverse_normalizer = InverseNormalizer(lang='en')

def text_normalization(original_text):
    text= clean(original_text,
        fix_unicode=True,               # fix various unicode errors
        to_ascii=True,                  # transliterate to closest ASCII representation
        lower=True,                     # lowercase text
        no_line_breaks=False,           # fully strip line breaks as opposed to only normalizing them
        no_urls=False,                  # replace all URLs with a special token
        no_emails=False,                # replace all email addresses with a special token
        no_phone_numbers=False,         # replace all phone numbers with a special token
        no_numbers=False,               # replace all numbers with a special token
        no_digits=False,                # replace all digits with a special token
        no_currency_symbols=False,      # replace all currency symbols with a special token
        no_punct=False,                 # remove punctuations
        replace_with_punct="",          # instead of removing punctuations you may replace them
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
        replace_with_number="<NUMBER>",
        replace_with_digit="0",
        replace_with_currency_symbol="<CUR>",
        lang="en"                       # set to 'de' for German special handling
    )
    text=inverse_normalizer.inverse_normalize(text, verbose=False)
    text=text.lower()
    # A dictionary of contractions and their expanded forms, including "didn't"
    contractions = {
        "i'm": "i am", "don't": "do not", "can't": "cannot", "it's": "it is",
        "isn't": "is not", "he's": "he is", "she's": "she is", "that's": "that is",
        "what's": "what is", "where's": "where is", "there's": "there is",
        "who's": "who is", "how's": "how is", "i've": "i have", "you've": "you have",
        "we've": "we have", "they've": "they have", "i'd": "i would", "you'd": "you would",
        "he'd": "he would", "she'd": "she would", "we'd": "we would", "they'd": "they would",
        "i'll": "i will", "you'll": "you will", "he'll": "he will", "she'll": "she will",
        "we'll": "we will", "they'll": "they will", "didn't": "did not"
    }

    # Manually handle contractions
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)

    # Remaining rules are the same as previous implementation
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    fillers = ["hmm", "mm", "mhm", "mmm", "uh", "um"]
    filler_pattern = r'\b(?:' + '|'.join(fillers) + r')\b'
    text = re.sub(filler_pattern, "", text)
    text = re.sub(r"\s’", "’", text)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"\.(?!\d)", "", text)
    text = re.sub(r"[^\w\s.,%$]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()
