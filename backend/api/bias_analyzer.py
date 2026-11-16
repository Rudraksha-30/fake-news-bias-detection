import re
LOADED_WORDS = {
    "shocking", "disgraceful", "outrageous", "radical", "extremist", "corrupt",
    "devastating", "slams", "destroys", "scandal", "evil", "disaster", "traitor",
    "propaganda", "regime", "puppet", "brainwashed", "sinister", "lunatic",
    "ridiculous", "pathetic", "failed", "betrayal", "catastrophic", "horrific",
    "heartless", "dangerous", "unbelievable", "exposed", "bombshell",
}
ABSOLUTES = {
    "always", "never", "everyone", "nobody", "all", "none", "totally",
    "completely", "entirely", "undeniable", "undeniably", "obviously",
    "clearly", "absolutely", "every", "nothing", "everything", "proven",
}
INTENSIFIERS = {"very", "extremely", "incredibly", "utterly", "massively", "hugely", "so", "really", "truly", "wildly"}
WEIGHTS = {"loaded": 0.30, "absolutes": 0.15, "intensifiers": 0.10, "caps_words": 0.10, "exclamations": 0.10}
MODEL_WEIGHT = 0.65; LEXICON_WEIGHT = 0.35; FLAG_THRESHOLD = 0.30; HIGH_AT = 0.45; MODERATE_AT = 0.20
MIN_WORDS = 4; MAX_SENTENCES = 40

class BiasAnalyzer:
    def __init__(self, use_model=False): self.use_model = use_model
    @staticmethod
    def split_sentences(text):
        text = re.sub(r"\s+", " ", text.strip())
        parts = re.split(r"(?<=[.!?])\s+", text)
        merged, carry = [], ""
        for p in parts:
            p = (carry + " " + p).strip() if carry else p
            if len(p.split()) < MIN_WORDS: carry = p
            else: merged.append(p); carry = ""
        if carry and merged: merged[-1] += " " + carry
        return merged[:MAX_SENTENCES]
    @staticmethod
    def _lexicon_score(sentence):
        words = re.findall(r"[A-Za-z']+", sentence); lower = [w.lower() for w in words]
        t = {"loaded": sorted({w for w in lower if w in LOADED_WORDS}),
             "absolutes": sorted({w for w in lower if w in ABSOLUTES}),
             "intensifiers": sorted({w for w in lower if w in INTENSIFIERS}),
             "caps_words": sorted({w for w in words if len(w) >= 3 and w.isupper()}),
             "exclamations": ["!"] * sentence.count("!")}
        return min(1.0, sum(WEIGHTS[k] * len(v) for k, v in t.items())), t
    def analyze(self, text):
        ss = self.split_sentences(text)
        if not ss: return {"available": False}
        sc, fl = [], []
        for s in ss:
            x, t = self._lexicon_score(s); sc.append(x)
            if x >= FLAG_THRESHOLD: fl.append((s, round(x, 3), {k: v for k, v in t.items() if v}))
        overall = 0.6 * sum(sc)/len(sc) + 0.4 * max(sc)
        label = "HIGH" if overall >= HIGH_AT else "MODERATE" if overall >= MODERATE_AT else "LOW"
        return {"available": True, "score": round(overall, 3), "label": label, "n": len(ss), "flagged": fl}
