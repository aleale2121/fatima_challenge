"""Helpers for splitting Ethiopic and Hangul syllables into consonant + vowel.

Unicode stores Ethiopic in rows of 8 (seven vowel orders, then the labialised
form), so row = consonant and column = vowel. Hangul is 0xAC00 + (lead*21 + vowel)*28 + tail.
"""

ETH_START, ETH_END = 0x1200, 0x135A  # punctuation starts at 0x1360

# the 33 Amharic base consonants (1st order); 33 x 7 = 231 core characters
ETH_BASES = list("ሀለሐመሠረሰሸቀበተቸኀነኘአከኸወዐዘዠየደጀገጠጨጰጸፀፈፐ")
assert len(ETH_BASES) == 33

# ግዕዝ ካዕብ ሣልስ ራብዕ ኃምስ ሳድስ ሳብዕ
ETH_ORDERS = ["ä", "u", "i", "a", "e", "ə", "o"]
ETH_ORDER_NAMES = ["gəʿəz", "kaʿəb", "saləs", "rabəʿ", "haməs", "sadəs", "sabəʿ"]


def eth_compose(base: str, order: int) -> str:
    """base is the 1st-order form, order is 1-7."""
    return chr(ord(base) + order - 1)


def eth_core_chars():
    """All 231 core characters as (char, base, order)."""
    return [(eth_compose(b, o), b, o) for b in ETH_BASES for o in range(1, 8)]


def is_ethiopic(ch: str) -> bool:
    return ETH_START <= ord(ch) < ETH_END


HANGUL_LEADS = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
# seven vowels so the Korean control has the same 7-way choice as Fidel
HANGUL_VOWELS = [(0, "ㅏ", "a"), (4, "ㅓ", "eo"), (8, "ㅗ", "o"), (13, "ㅜ", "u"),
                 (18, "ㅡ", "eu"), (20, "ㅣ", "i"), (1, "ㅐ", "ae")]


def is_hangul(ch: str) -> bool:
    return 0xAC00 <= ord(ch) <= 0xD7A3


def hangul_compose(lead: int, vowel: int, tail: int = 0) -> str:
    return chr(0xAC00 + (lead * 21 + vowel) * 28 + tail)


def hangul_core_chars():
    """19 consonants x 7 vowels, no final consonant."""
    out = []
    for li, lj in enumerate(HANGUL_LEADS):
        for k, (vi, _, _) in enumerate(HANGUL_VOWELS):
            out.append((hangul_compose(li, vi), lj, k + 1))
    return out


def decompose(ch: str):
    """(script, consonant, vowel) for a syllable, None for anything else.

    For Hangul the "consonant" is lead + final, so two syllables that only
    differ in the vowel count as a vowel-form mistake.
    """
    cp = ord(ch)
    if is_ethiopic(ch):
        off = cp - ETH_START
        return ("eth", off // 8, off % 8)
    if is_hangul(ch):
        off = cp - 0xAC00
        lead, rem = divmod(off, 21 * 28)
        vowel, tail = divmod(rem, 28)
        return ("hangul", (lead, tail), vowel)
    return None


def substitution_type(ref: str, hyp: str) -> str:
    """What kind of mistake is reading ref as hyp?"""
    if ref == hyp:
        return "correct"
    dr, dh = decompose(ref), decompose(hyp)
    if dr is None:
        return "other"
    if dh is None and 0x3131 <= ord(hyp) <= 0x318E:
        return "fragment"  # a lone jamo like ㅡ instead of a full syllable
    if dh is None or dh[0] != dr[0]:
        return "script_escape"  # wrote Latin, Greek, Georgian, ...
    if dr[1] == dh[1]:
        return "vowel_form"
    if dr[2] == dh[2]:
        return "consonant"
    return "both"


def align(ref: str, hyp: str):
    """Plain Levenshtein alignment -> [(op, ref_char, hyp_char), ...]"""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("eq" if ref[i - 1] == hyp[j - 1] else "sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", hyp[j - 1]))
            j -= 1
    return ops[::-1]


def error_breakdown(ref: str, hyp: str):
    """Count edits, splitting substitutions by vowel / consonant / script."""
    counts = {"ref_len": len(ref), "eq": 0, "del": 0, "ins": 0, "vowel_form": 0,
              "consonant": 0, "both": 0, "script_escape": 0, "fragment": 0, "other": 0}
    for op, r, h in align(ref, hyp):
        if op == "sub":
            counts[substitution_type(r, h)] += 1
        else:
            counts[op] += 1
    return counts
