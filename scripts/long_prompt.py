"""Build one long single-line prompt for hand-testing long context (scripts/chat_repl.py reads one line per prompt).

  passkey  a 5-digit pass key hidden in filler text            -> "...The pass key is"           expect the key
  fact     a fact stated near the start, asked at the end       -> "...Mira's dog is named"        expect Biscuit
  twohop   two facts far apart that must be chained             -> "...lives in"                   expect Lisbon

--repeat sets the length (each filler unit is ~22 tokens: 350 ≈ 8k tokens, 650 ≈ 15k); --depth places the hidden text
(0 = start, 1 = end).

    python3 scripts/long_prompt.py --kind passkey --repeat 650 --depth 0.5 > /tmp/lp.txt
    python3 scripts/long_prompt.py --kind passkey --repeat 650 --chat > /tmp/lp_chat.txt   # question form, for /chat + /file
"""
import argparse

FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["passkey", "fact", "twohop"], default="passkey")
    ap.add_argument("--repeat", type=int, default=350)
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--key", default="71432")
    ap.add_argument("--chat", action="store_true", help="end with a plain question (for chat mode / the SFT model)")
    a = ap.parse_args()
    n = a.repeat
    cut = int(n * a.depth)
    if a.kind == "passkey":
        head = "There is an important pass key hidden inside a lot of irrelevant text. Find it and remember it. "
        needle = f"The pass key is {a.key}. Remember it. {a.key} is the pass key. "
        tail = "What is the pass key?" if a.chat else "What is the pass key? The pass key is"
        text = head + FILLER * cut + needle + FILLER * (n - cut) + tail
    elif a.kind == "fact":
        text = ("Mira has a small brown dog named Biscuit. " + FILLER * n +
                ("What is the name of Mira's dog?" if a.chat else "Earlier we learned about Mira and her dog. Mira's dog is named"))
    else:
        text = ("The Silomar bakery is owned by Kwame. " + FILLER * cut + "Kwame lives in Lisbon. " +
                FILLER * (n - cut) + ("Where does the owner of the Silomar bakery live?" if a.chat else "The owner of the Silomar bakery lives in"))
    print(text)


if __name__ == "__main__":
    main()
