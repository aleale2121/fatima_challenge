"""Builds the probe set: probes.jsonl plus the rendered images.

Gold answers come from the Unicode layout, so nothing is hand-labelled.
Each Ethiopic task has a Korean twin to compare against.
"""
import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from scripts import (ETH_BASES, HANGUL_LEADS, HANGUL_VOWELS, decompose,
                     eth_compose, eth_core_chars, hangul_compose,
                     hangul_core_chars, is_ethiopic)

ETH_FONT = "/usr/share/fonts/truetype/noto/NotoSansEthiopic-Regular.ttf"
ETH_FONT_ALT = "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-Regular.ttf"
KOR_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

# everyday Amharic words
AMHARIC_WORDS = """ሰላም ኢትዮጵያ አዲስ አበባ ቤት ውሃ ዳቦ ቡና ትምህርት መጽሐፍ ልጅ እናት አባት ወንድም
እህት ሀገር ከተማ መንገድ መኪና ገበያ ገንዘብ ስራ ጊዜ ቀን ሌሊት ፀሐይ ጨረቃ ኮከብ ዝናብ ተራራ
ወንዝ ባሕር ዛፍ ፍቅር ሰው ሴት ወንድ ቤተሰብ ምግብ እንጀራ ወጥ ሻይ ወተት ማር ጨው በርበሬ
ሽንኩርት ዶሮ በግ ላም ፈረስ ውሻ ድመት ወፍ አሳ ታሪክ ሙዚቃ ዘፈን ጨዋታ ኳስ ተማሪ መምህር
ሐኪም ሆስፒታል ቢሮ ስልክ ደብዳቤ ጥያቄ መልስ ቁጥር ፊደል ቃል ዓመት ወር ሳምንት ዛሬ ነገ
ትናንት ጥሩ መጥፎ ትልቅ ትንሽ ቆንጆ ሞቃት ቀዝቃዛ አመሰግናለሁ እንኳን ደህና ነጻነት ሰላምታ
ዩኒቨርሲቲ ቤተክርስቲያን መስጊድ ቴሌቪዥን ሬዲዮ ጋዜጣ""".split()

KOREAN_WORDS = """안녕 사랑 학교 친구 가족 어머니 아버지 나무 바다 하늘 사람 시간 음식 김치
한국 서울 도시 자동차 컴퓨터 전화 노래 음악 영화 책 공부 선생님 학생 병원 의사 물 우유
커피 고양이 강아지 새 꽃 봄 여름 가을 겨울 아침 저녁 오늘 내일 어제 좋다 나쁘다 크다
작다 감사합니다 미안합니다 대학교 도서관 시장 돈 일 이름 나라 언어 역사 바람 구름 비
눈 산 강 길 문 집 방 창문 의자 신문 편지 질문 대답 숫자 글자 단어 주 달 해""".split()

# short sentences for the line OCR task
AMHARIC_LINES = [
    "ሰላም እንዴት ነህ?", "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።", "ቡና መጠጣት እወዳለሁ።",
    "ልጆቹ ትምህርት ቤት ሄዱ።", "ዛሬ ዝናብ ይዘንባል።", "እናቴ እንጀራ ጋገረች።",
    "መጽሐፉን አነበብኩት።", "ነገ ገበያ እንሄዳለን።", "ውሻው በግቢው ውስጥ ይጫወታል።",
    "ፀሐይ በምሥራቅ ትወጣለች።", "አባቴ መኪና ይነዳል።", "ተማሪዎቹ ፈተና ወሰዱ።",
    "ሐኪሙ ታማሚውን አከመ።", "ወንዙ በጣም ትልቅ ነው።", "ስልኬ ጠፋብኝ።",
    "እህቴ ዘፈን ትዘፍናለች።", "ጓደኛዬ ደብዳቤ ጻፈልኝ።", "ቤተሰቤ በጣም ትልቅ ነው።",
    "ሻይ ወይስ ቡና ትፈልጋለህ?", "የአማርኛ ፊደል ብዙ ቁምፊዎች አሉት።", "ዶሮዎቹ በጠዋት ይጮኻሉ።",
    "መምህሩ ጥያቄ ጠየቀ።", "ገንዘቤን በባንክ አስቀመጥኩ።", "በዓሉ በደስታ ተከበረ።",
    "ሙዚቃ ማዳመጥ ያስደስተኛል።", "ተራራውን ወጣን።", "ጨረቃዋ ዛሬ ሙሉ ናት።",
    "ኳስ መጫወት እንወዳለን።", "ቀኑ በጣም ሞቃት ነው።", "አመሰግናለሁ ደህና ሁን።",
]

KOREAN_LINES = [
    "안녕하세요 만나서 반갑습니다.", "서울은 한국의 수도입니다.", "저는 커피를 좋아합니다.",
    "아이들이 학교에 갔습니다.", "오늘 비가 옵니다.", "어머니가 김치를 만드셨다.",
    "그 책을 다 읽었습니다.", "내일 시장에 갑니다.", "강아지가 마당에서 놉니다.",
    "해가 동쪽에서 뜹니다.", "아버지가 차를 운전합니다.", "학생들이 시험을 봤습니다.",
    "의사가 환자를 치료했다.", "그 강은 아주 큽니다.", "전화기를 잃어버렸어요.",
    "언니가 노래를 부릅니다.", "친구가 편지를 써 주었다.", "우리 가족은 대가족입니다.",
    "차와 커피 중 무엇을 원하세요?", "한글은 자음과 모음으로 이루어져 있다.",
    "닭이 아침에 웁니다.", "선생님이 질문을 했다.", "은행에 돈을 맡겼습니다.",
    "명절을 즐겁게 보냈다.", "음악 듣는 것을 좋아해요.", "우리는 산에 올랐다.",
    "오늘 밤 달이 둥글다.", "축구 하는 것을 좋아합니다.", "날씨가 매우 덥습니다.",
    "감사합니다 안녕히 가세요.",
]

ETH_SCHEME = ("The Ethiopic (Ge'ez/Fidel) script writes each syllable as a consonant in one of "
              "seven vowel orders: 1st=ä, 2nd=u, 3rd=i, 4th=a, 5th=e, 6th=ə (or no vowel), 7th=o. "
              "For example, the consonant በ (b) has the forms በ(bä) ቡ(bu) ቢ(bi) ባ(ba) ቤ(be) ብ(bə) ቦ(bo).")
KOR_SCHEME = ("The Korean Hangul script writes each syllable as a consonant combined with a vowel. "
              "In this question we number seven vowels as: 1st=ㅏ(a), 2nd=ㅓ(eo), 3rd=ㅗ(o), 4th=ㅜ(u), "
              "5th=ㅡ(eu), 6th=ㅣ(i), 7th=ㅐ(ae). For example, the consonant ㅂ (b) has the forms "
              "바(ba) 버(beo) 보(bo) 부(bu) 브(beu) 비(bi) 배(bae).")
ORD = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th"]
ONE_CHAR = "Answer with a single character only."


def kor_order_char(lead_jamo, order):
    return hangul_compose(HANGUL_LEADS.index(lead_jamo), HANGUL_VOWELS[order - 1][0])


def text_char_probes(rng):
    items = []
    sets = [("eth", eth_core_chars(), ETH_SCHEME), ("hangul", hangul_core_chars(), KOR_SCHEME)]
    for script, chars, scheme in sets:
        for ch, base, order in chars:
            meta = {"script": script, "char": ch, "base": base, "order": order}
            base_word = "1st-order (base) form" if script == "eth" else "consonant letter (jamo)"
            items.append({**meta, "task": "decompose_consonant",
                          "prompt": f"{scheme}\n\nWhat is the base consonant of the syllable {ch}? "
                                    f"Give its {base_word}. {ONE_CHAR}",
                          "gold": base, "answer_type": "char"})
            items.append({**meta, "task": "identify_order",
                          "prompt": f"{scheme}\n\nWhich vowel order (1-7) is the syllable {ch}? "
                                    "Answer with a single digit only.",
                          "gold": str(order), "answer_type": "digit"})
            tgt = rng.choice([o for o in range(1, 8) if o != order])
            compose = eth_compose(base, tgt) if script == "eth" else kor_order_char(base, tgt)
            items.append({**meta, "task": "compose", "target_order": tgt,
                          "prompt": f"{scheme}\n\nWrite the {ORD[tgt - 1]}-order syllable of the "
                                    f"consonant {base}. {ONE_CHAR}",
                          "gold": compose, "answer_type": "char"})
            items.append({**meta, "task": "transform", "target_order": tgt,
                          "prompt": f"{scheme}\n\nKeep the consonant of the syllable {ch} but change it "
                                    f"to the {ORD[tgt - 1]} vowel order. {ONE_CHAR}",
                          "gold": compose, "answer_type": "char"})
    return items


def flip(word, script, kind, rng):
    """Swap the vowel or the consonant of one random syllable in the word."""
    base_rows = [(ord(b) - 0x1200) // 8 for b in ETH_BASES]
    idx = [i for i, c in enumerate(word) if decompose(c) and
           (script != "eth" or ((ord(c) - 0x1200) // 8 in base_rows and (ord(c) - 0x1200) % 8 < 7))]
    if not idx:
        return None
    i = rng.choice(idx)
    c = word[i]
    if script == "eth":
        off = ord(c) - 0x1200
        row, v = divmod(off, 8)
        if kind == "vowel":
            new = chr(0x1200 + row * 8 + rng.choice([o for o in range(7) if o != v]))
        else:
            new = chr(0x1200 + rng.choice([r for r in base_rows if r != row]) * 8 + v)
    else:
        off = ord(c) - 0xAC00
        lead, rem = divmod(off, 21 * 28)
        vowel, tail = divmod(rem, 28)
        if kind == "vowel":
            new = hangul_compose(lead, rng.choice([x for x in range(21) if x != vowel]), tail)
        else:
            new = hangul_compose(rng.choice([x for x in range(19) if x != lead]), vowel, tail)
    return word[:i] + new + word[i + 1:]


def minimal_pair_probes(rng):
    items = []
    for script, words in [("eth", AMHARIC_WORDS), ("hangul", KOREAN_WORDS)]:
        for w in words:
            for cond in ["identical", "vowel", "consonant"]:
                w2 = w if cond == "identical" else flip(w, script, cond, rng)
                if w2 is None:
                    continue
                a, b = (w, w2) if rng.random() < 0.5 else (w2, w)
                items.append({"script": script, "task": "minimal_pair", "condition": cond,
                              "word_a": a, "word_b": b,
                              "prompt": "Are these two words spelled exactly the same, character for "
                                        f"character?\nWord A: {a}\nWord B: {b}\nAnswer yes or no.",
                              "gold": "yes" if cond == "identical" else "no",
                              "answer_type": "yesno"})
    return items


def render(text, font_path, size, out_path, pad=24):
    font = ImageFont.truetype(font_path, size)
    l, t, r, b = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), text, font=font)
    img = Image.new("RGB", (r - l + 2 * pad, b - t + 2 * pad), "white")
    ImageDraw.Draw(img).text((pad - l, pad - t), text, font=font, fill="black")
    img.save(out_path)


def visual_probes(out):
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for script, chars, font in [("eth", eth_core_chars(), ETH_FONT),
                                ("hangul", hangul_core_chars(), KOR_FONT)]:
        for ch, base, order in chars:
            name = f"char_{script}_{ord(ch):04X}.png"
            render(ch, font, 64, img_dir / name)
            items.append({"script": script, "task": "ocr_char", "char": ch, "base": base,
                          "order": order, "image": f"images/{name}",
                          "prompt": "What single character is shown in this image? "
                                    "Reply with only that character.",
                          "gold": ch, "answer_type": "text"})
    for script, words, lines, font in [("eth", AMHARIC_WORDS, AMHARIC_LINES, ETH_FONT),
                                       ("hangul", KOREAN_WORDS, KOREAN_LINES, KOR_FONT)]:
        for i, w in enumerate(words):
            name = f"word_{script}_{i:03d}.png"
            render(w, font, 48, img_dir / name)
            items.append({"script": script, "task": "ocr_word", "image": f"images/{name}",
                          "prompt": "Transcribe the text in this image exactly. "
                                    "Reply with only the transcription.",
                          "gold": w, "answer_type": "text"})
        for i, s in enumerate(lines):
            name = f"line_{script}_{i:03d}.png"
            render(s, font, 40, img_dir / name)
            items.append({"script": script, "task": "ocr_line", "image": f"images/{name}",
                          "prompt": "Transcribe the text in this image exactly. "
                                    "Reply with only the transcription.",
                          "gold": s, "answer_type": "text"})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    items = text_char_probes(rng) + minimal_pair_probes(rng) + visual_probes(out)
    for i, it in enumerate(items):
        it["id"] = f"{it['task']}-{it['script']}-{i:05d}"
    with open(out / "probes.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    from collections import Counter
    print(Counter((it["task"], it["script"]) for it in items))
    assert all(is_ethiopic(c) or not c.strip() or c in "?።" for w in AMHARIC_WORDS for c in w)


if __name__ == "__main__":
    main()
