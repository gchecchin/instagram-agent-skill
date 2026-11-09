#!/usr/bin/env python3
"""
hookscore.py - score the first line of a Reel on the five things strong hooks
have in common, and rank a batch of them against each other.

What this is:  five local heuristics, computed on your machine from the text
alone. They measure properties that hooks which hold attention tend to share -
a length you can say in under three seconds, a concrete marker, something at stake,
the payload at the front rather than the back, and a viewer to aim it at.

What this is NOT:  a view predictor. It was tested against 74 real short-form
hooks, transcribed from the first three seconds of the top eight and bottom
eight performers on five channels. Separating a real hook from a deliberately
bad one it does well: AUC 0.83, and 9 of 10 written-to-be-bad hooks scored
below the real median. Separating a good creator's hits from that same
creator's misses it barely does at all: AUC 0.56, where 0.50 is a coin flip.

So use it for what it measured well. It catches greetings, preambles, hooks
with nothing concrete in them and hooks that take five seconds to say. It will
not tell you which of two decent hooks will travel, and nothing that reads text
can, because that is decided by your face, your edit, your audio and who
Instagram shows it to. Trust the retention graph over this script.

Each check returns 0 to 100. Higher is better.

Usage
  python3 hookscore.py hooks.txt              # one hook per line, ranked
  python3 hookscore.py --hook "I lost $18,000 on one missing contract."
  pbpaste | python3 hookscore.py -
  python3 hookscore.py hooks.txt --json
"""

import argparse
import json
import re
import statistics
import sys

WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9$€%'’-]+")
NUMBER_RE = re.compile(
    r"[$€]\s?\d[\d,]*(?:\.\d+)*"                     # money, whole
    r"|\b\d[\d,]*(?:\.\d+)*\s?"                      # a figure, with or
    r"(?:%|k\b|x\b|hrs?\b|hours?\b|mins?\b|minutes?\b"  # without a unit
    r"|days?\b|weeks?\b|months?\b|years?\b"
    r"|€|euro\b|or[ae]\b|minut[oi]\b|giorn[oi]\b|settiman[ae]\b|mes[ei]\b"
    r"|ann[oi]\b|volte\b|sedute\b)?",
    re.IGNORECASE)
PROPER_RE = re.compile(r"(?<!^)\b[A-Z][a-zà-öø-ÿ]{2,}\b")
HASHTAG_RE = re.compile(r"(?:^|\s)#\w+")
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]")

# Spoken hooks say their numbers out loud. "Zero dollars" and "twenty grand"
# are as concrete as "$0" and "$20,000", and counting only digits missed them.
# "one" and "first" are deliberately absent: they are filler far more often
# than they are a quantity.
SPOKEN_NUMBERS = {
    "zero", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "fifteen", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
    "billion", "dozen", "half", "twice", "triple",
}
MONEY_WORDS = {
    "dollars", "dollar", "bucks", "grand", "percent", "cents",
    "millionaire", "billionaire", "revenue", "profit", "salary", "rent",
}

# Words that put something on the line. A hook with none of these is a
# statement; a hook with one is a reason to keep watching.
STAKES = {
    "stop", "never", "wrong", "mistake", "mistakes", "lost", "lose", "losing",
    "cost", "costs", "broke", "broken", "failed", "failure", "fail", "nobody",
    "no", "not", "don't", "dont", "doesn't", "didn't", "can't", "won't",
    "quit", "quitting", "fired", "deleted", "delete", "killed", "kills", "kill",
    "replaced", "replaces", "cut", "beat", "free", "paid", "charged", "hired",
    "saved", "first",
    "banned", "illegal", "worst", "hate", "hated", "wasted", "waste", "scam",
    "lie", "lied", "lying", "truth", "secret", "hidden", "stole", "stolen",
    "before", "until", "instead", "but", "except", "unless", "problem",
    "risk", "danger", "warning", "regret", "wish", "should", "shouldn't",
    "still", "already", "only", "without", "versus", "vs", "actually",
}

# Openers that spend the first second saying nothing.
WEAK_OPENERS = [
    "so", "ok", "okay", "hey", "hi", "hello", "guys", "yo", "alright",
    "welcome", "today", "basically", "honestly", "look", "listen", "um",
    "just", "let", "lets", "let's", "i wanted", "i want", "one of",
    "have you", "did you", "do you", "are you", "in this", "in today",
    "the thing", "a lot", "there is", "there are", "this is", "it is",
    "as a", "when it", "if you've", "you know",
]

# Imperatives that earn the front position.
IMPERATIVES = {
    "stop", "steal", "copy", "delete", "try", "watch", "read", "save",
    "use", "build", "make", "write", "send", "take", "start", "quit",
    "never", "always", "don't", "dont", "do", "put", "run", "check",
}

# Italian. Same five properties, same weights; only the vocabulary changes.
# Number words compound ("duemila", "quarantacinque", "ventitré"), so they are
# matched by shape rather than listed. "uno" and "una" stay out for the reason
# "one" does, and "sei" is also "you are": it counts as six only when a unit
# follows it ("sei mesi", not "sei sicura").
IT_NUMBER_RE = re.compile(
    r"(?:due|tre|tré|quattro|cinque|sei|sette|otto|nove|dieci|undici|dodici"
    r"|tredici|quattordici|quindici|sedici|diciassette|diciotto|diciannove"
    r"|(?:vent|trent|quarant|cinquant|sessant|settant|ottant|novant)[aio]?"
    r"|cento|mille|mila|uno?|milion[ei]|miliard[oi])+")
IT_UNITS = {
    "mesi", "anni", "giorni", "ore", "minuti", "secondi", "settimane", "euro",
    "volte", "sedute", "pazienti", "clienti", "persone", "punti", "chili",
    "kg", "cm", "ml", "fiale", "siringhe",
}
# Italian elides articles and prepositions onto the next word ("l'acido",
# "dell'anno", "c'è"). The word that carries meaning is the one after the
# apostrophe. English contractions ("don't", "it's") keep theirs.
ELISION_RE = re.compile(
    r"^(?:l|un|dell|nell|all|dall|sull|coll|quest|quell|nessun|tutt|c|d|m|n|s|t|v)"
    r"['’](.+)$")

SPOKEN_NUMBERS |= {
    "zero", "mezzo", "mezza", "metà", "doppio", "triplo", "dozzina",
    "decine", "centinaia", "migliaia",
}
MONEY_WORDS |= {
    "euro", "soldi", "percento", "centesimi", "milionario", "fatturato",
    "guadagno", "guadagni", "profitto", "stipendio", "affitto",
}
STAKES |= {
    "smetti", "smettila", "basta", "mai", "sbagliato", "sbagliata", "sbagliati",
    "sbagliate", "errore", "errori", "sbaglio", "perso", "persa", "persi",
    "perdere", "perdi", "costa", "costano", "costato", "costata", "costo",
    "rovinato", "rovinata", "fallito", "fallita", "nessuno", "nessuna",
    "niente", "nulla", "non", "neanche", "nemmeno", "licenziato", "cancellato",
    "eliminato", "sostituito", "tagliato", "battuto", "gratis", "pagato",
    "pagata", "risparmiato", "assunto", "primo", "prima", "vietato", "illegale",
    "peggiore", "peggio", "odio", "odiavo", "sprecato", "sprecati", "spreco",
    "buttato", "buttata", "buttati", "buttate",
    "truffa", "bugia", "mentito", "verità", "segreto", "nascosto", "rubato",
    "finché", "invece", "ma", "però", "tranne", "senza", "problema", "rischio",
    "rischi", "pericolo", "pericoloso", "attenzione", "rimpianto", "vorrei",
    "dovresti", "ancora", "già", "solo", "soltanto", "contro", "davvero",
    "evita", "evitare",
}
WEAK_OPENERS += [
    "allora", "ciao", "ragazzi", "buongiorno", "buonasera", "salve", "ehi",
    "eccoci", "benvenuti", "benvenute", "oggi", "praticamente", "sinceramente",
    "onestamente", "dunque", "insomma", "cioè", "beh", "ecco", "ascolta",
    "ascoltate", "senti", "sentite", "volevo", "voglio", "vorrei", "vi voglio",
    "in questo", "nel video", "hai mai", "avete mai", "ti sei", "vi siete",
    "lo sai", "sapevi", "sapete", "c'è", "ci sono", "questo è", "questa è",
    "una delle", "uno dei", "molti", "molte", "tanti", "tante", "se anche",
    "quando si",
]
IMPERATIVES |= {
    "smetti", "smettila", "ruba", "copia", "cancella", "prova", "guarda",
    "leggi", "salva", "usa", "costruisci", "scrivi", "manda", "prendi",
    "inizia", "metti", "controlla", "evita", "dimentica", "chiedi",
    "mai", "sempre", "fai", "fallo", "provalo", "salvalo", "copiala",
}
IT_YOU = {
    "tu", "ti", "te", "tuo", "tua", "tuoi", "tue", "voi", "vi", "vostro",
    "vostra", "vostri", "vostre", "hai", "sei", "fai", "puoi", "devi", "sai",
    "vuoi", "avete", "siete", "fate", "potete", "dovete", "sapete", "volete",
}
IT_ME = {
    "io", "mi", "me", "mio", "mia", "miei", "mie", "noi", "ci", "nostro",
    "nostra", "nostri", "nostre", "ho", "abbiamo", "sono",
}

DEALBREAKERS = [
    (re.compile(r"(?i)^\s*(?:stop scrolling|don'?t scroll"
                r"|smetti(?:la)? di scorrere|non scorrere|non scrollare|fermati\b)"),
     "Opens with \"stop scrolling\". Asking for attention proves you have not earned it."),
    (re.compile(r"(?i)\b(?:in (?:this|today'?s) (?:video|reel)|i'?m going to show you|i'?ll show you how"
                r"|in questo (?:video|reel)|nel (?:video|reel) di oggi"
                r"|(?:vi|ti) (?:mostro|faccio vedere|spiego|parlo)"
                r"|(?:vi|ti) voglio (?:mostrare|parlare|spiegare))\b"),
     "Video preamble. Delete it and open on the payoff."),
    (re.compile(r"(?i)^\s*(?:hey |hi |what'?s up |welcome "
                r"|ciao\b|buongiorno\b|buonasera\b|salve\b|ehi\b|benvenut[ie]\b)"),
     "Greeting. Nobody came to the feed to be greeted."),
    (HASHTAG_RE,
     "Hashtag in the hook. Hashtags belong at the bottom of the caption, if anywhere."),
    (EMOJI_RE,
     "Emoji in the hook. On-screen text at hook size has room for words or for an emoji, not both."),
]


def clamp(n):
    return max(0.0, min(100.0, n))


def words(text):
    # "$18,000" is one word when it is spoken, so it is one word here too.
    # So is "4.000 €", which is how Italian writes the thousands.
    return WORD_RE.findall(re.sub(r"(?<=\d),(?=\d)|(?<=\d)\.(?=\d{3}\b)", "", text))


def terms(text):
    """words(), lower-cased, with an Italian elision resolved to its word."""
    out = []
    for token in words(text):
        t = token.lower().strip("'’")
        m = ELISION_RE.match(t)
        out.append(m.group(1) if m else t)
    return out


def spoken_number(low, i):
    """Is low[i] a number said out loud, in English or in Italian."""
    t = low[i]
    if t in SPOKEN_NUMBERS:
        return True
    if t in ("un", "uno", "una") or not IT_NUMBER_RE.fullmatch(t):
        return False
    return t != "sei" or (i + 1 < len(low) and low[i + 1] in IT_UNITS)


def check_length(text):
    """A hook has to land before the thumb moves. Roughly two seconds."""
    w = words(text)
    n = len(w)
    secs = n / 2.75                      # ~165 words per minute, spoken
    chars = len(text.strip())
    if 5 <= n <= 12:
        score = 100.0
    elif n < 5:
        score = clamp(100 - (5 - n) * 20)
    else:
        score = clamp(100 - (n - 12) * 11)
    if chars > 60:                       # two lines of big on-screen text
        score -= 12
    return clamp(score), f"{n} words, {chars} chars, ~{secs:.1f}s spoken (want 5-12 words)"


def check_specificity(text):
    """One concrete thing beats three abstract ones."""
    nums = [n.strip() for n in NUMBER_RE.findall(text) if n.strip()]
    propers = set(PROPER_RE.findall(text))
    low = terms(text)
    spoken = [w for i, w in enumerate(low) if spoken_number(low, i) or w in MONEY_WORDS]
    hits = len(nums) + len(propers) + len(spoken)
    score = 15.0 if hits == 0 else clamp(45 + hits * 30)
    found = ", ".join(nums[:2] + sorted(propers)[:2] + spoken[:2])
    return score, (f"{hits} concrete marker(s)" + (f": {found}" if found else
                   " - no number, no name, nothing checkable"))


def check_stakes(text):
    """Tension, cost, negation. Something the viewer might lose."""
    w = terms(text)
    hits = [x for x in w if x in STAKES]
    markers = sorted(set(hits))
    if re.search(r"[$€]\s?\d|\d\s?(?:€|euro\b)", text, re.IGNORECASE):
        markers.append("a price")
    n = len(markers)
    score = {0: 20.0, 1: 70.0}.get(n, 100.0)
    detail = f"{n} tension marker(s)" + (f": {', '.join(markers[:4])}" if markers else
                                         " - nothing is at stake in this line")
    return clamp(score), detail


def check_frontload(text):
    """The interesting word cannot be in position nine."""
    w = words(text)
    if not w:
        return 0.0, "empty"
    raw = [x.lower().strip("'’") for x in w]
    low = terms(text)
    opener = " ".join(raw[:2])
    penalty = 0
    hit_opener = None
    for weak in WEAK_OPENERS:
        # Whole words only: "so" is not the start of "Solo" or "Someone".
        if weak in (raw[0], opener):
            penalty, hit_opener = 30, weak
            break
    payload = None
    for i, token in enumerate(low):
        if (token in STAKES or spoken_number(low, i) or token in MONEY_WORDS
                or NUMBER_RE.match(w[i]) or (i and PROPER_RE.match(w[i]))):
            payload = i
            break
    if payload is None:
        base = 30.0
        where = "no payload word anywhere in the line"
    elif payload <= 3:
        base = 100.0
        where = f"payload at word {payload + 1}"
    elif payload <= 6:
        base = 70.0
        where = f"payload at word {payload + 1}, could move forward"
    else:
        base = 40.0
        where = f"payload at word {payload + 1}, too late"
    detail = where + (f"; weak opener \"{hit_opener}\"" if hit_opener else "")
    return clamp(base - penalty), detail


def check_address(text):
    """Aimed at one viewer, or floating in the air."""
    low = text.lower()
    w = terms(text)
    if (re.search(r"\b(you|your|you're|youre|yourself)\b", low)
            or any(t in IT_YOU and not spoken_number(w, i) for i, t in enumerate(w))):
        return 100.0, "speaks to the viewer"
    if w and w[0] in IMPERATIVES:
        return 90.0, f"imperative opener (\"{w[0]}\")"
    # Italian says "don't" as "non" plus the infinitive: "Non fare il filler".
    if len(w) > 1 and w[0] == "non" and re.search(r"(?:are|ere|ire)$", w[1]):
        return 90.0, f"imperative opener (\"non {w[1]}\")"
    if re.search(r"\b(i|my|me|we|our)\b", low) or any(t in IT_ME for t in w):
        return 70.0, "first person, no viewer named"
    return 35.0, "third person, nobody in the room"


CHECKS = ["LENGTH", "SPECIFICITY", "STAKES", "FRONTLOAD", "ADDRESS"]


def run(text):
    results = {
        "LENGTH": check_length(text),
        "SPECIFICITY": check_specificity(text),
        "STAKES": check_stakes(text),
        "FRONTLOAD": check_frontload(text),
        "ADDRESS": check_address(text),
    }
    flags = [msg for pattern, msg in DEALBREAKERS if pattern.search(text)]
    scores = [results[c][0] for c in CHECKS]
    # The weakest property caps the hook, same logic as detect.py: one bad
    # property is enough for the thumb to keep moving.
    overall = statistics.mean(scores) * 0.6 + min(scores) * 0.4 - len(flags) * 15
    overall = clamp(overall)
    verdict = "STRONG" if overall >= 70 and min(scores) >= 55 and not flags else (
        "OK" if overall >= 50 else "WEAK")
    return results, overall, verdict, flags


def bar(score, width=24):
    filled = round(score / 100 * width)
    return "#" * filled + "." * (width - filled)


def render_one(text, results, overall, verdict, flags, out=sys.stdout):
    print("\nHOOK SCORE", file=out)
    print("=" * 62, file=out)
    print(f"  \"{text.strip()}\"\n", file=out)
    for name in CHECKS:
        score, detail = results[name]
        print(f"  {name:<13} {bar(score)} {score:5.1f}", file=out)
        print(f"  {'':<13} {detail}", file=out)
    print("-" * 62, file=out)
    print(f"  {'HOOK SCORE':<13} {bar(overall)} {overall:5.1f}   {verdict}", file=out)
    for f in flags:
        print(f"\n  DEALBREAKER  {f}", file=out)
    if verdict != "STRONG":
        weakest = min(CHECKS, key=lambda c: results[c][0])
        print(f"\n  Weakest property: {weakest}. Fix that one and re-run.", file=out)
    print("", file=out)


def render_table(rows, out=sys.stdout):
    print("\nHOOK RANKING\n" + "=" * 78, file=out)
    for i, r in enumerate(rows, 1):
        mark = "->" if i == 1 else "  "
        hook = r["hook"] if len(r["hook"]) <= 62 else r["hook"][:59] + "..."
        print(f"{mark} {r['score']:5.1f} {r['verdict']:<7} {hook}", file=out)
        print(f"        weakest: {r['weakest']} ({r['checks'][r['weakest']]['score']:.0f})", file=out)
        for f in r["flags"]:
            print(f"        dealbreaker: {f}", file=out)
    print("\nShoot the top one. If the top one is under 50, none of these are the hook.\n",
          file=out)


def main():
    ap = argparse.ArgumentParser(description="Score a Reel hook on five properties.")
    ap.add_argument("input", nargs="?", default="-", help="file with one hook per line, or -")
    ap.add_argument("--hook", help="score a single hook given on the command line")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.hook:
        lines = [args.hook]
    else:
        raw = sys.stdin.read() if args.input == "-" else open(args.input, encoding="utf-8").read()
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if not lines:
        print("nothing to score", file=sys.stderr)
        sys.exit(2)

    payload = []
    for line in lines:
        results, overall, verdict, flags = run(line)
        payload.append({
            "hook": line,
            "checks": {k: {"score": round(v[0], 1), "detail": v[1]} for k, v in results.items()},
            "weakest": min(CHECKS, key=lambda c: results[c][0]),
            "flags": flags,
            "score": round(overall, 1),
            "verdict": verdict,
        })

    if args.json:
        print(json.dumps(payload if len(payload) > 1 else payload[0], indent=2, ensure_ascii=False))
        return

    if len(payload) == 1:
        results, overall, verdict, flags = run(lines[0])
        render_one(lines[0], results, overall, verdict, flags)
    else:
        render_table(sorted(payload, key=lambda r: -r["score"]))

    sys.exit(0 if max(p["score"] for p in payload) >= 70 else 1)


if __name__ == "__main__":
    main()
