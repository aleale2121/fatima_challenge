---
license: cc-by-4.0
language: [am, ko]
tags: [evaluation, ethiopic, amharic, ocr, vision-language, tokenization, low-resource]
---

# FidelProbe: small frontier models read Ethiopic as a bag of opaque symbols

**Author:** Alefew Yimam (Fordham University)
**Model evaluated:** [`Qwen/Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) (2B parameters)
**Contents:** evaluation set (2,592 items + 602 rendered images), code, Colab notebook, raw and scored predictions, figures.

---

## 1. The blind spot: vowel-form blindness in the Ethiopic script

Amharic, spoken by more than 58 million people, is written in the Ethiopic script (*Fidel*, ፊደል).
Fidel is an **abugida**: each character is a consonant written in one of seven *vowel orders*.
The 33 core consonants × 7 orders give the 231 core characters every Ethiopian child learns as a grid:

| | ä | u | i | a | e | ə | o |
|---|---|---|---|---|---|---|---|
| **b** | በ | ቡ | ቢ | ባ | ቤ | ብ | ቦ |
| **s** | ሰ | ሱ | ሲ | ሳ | ሴ | ስ | ሶ |

Forms of one consonant differ only by a small stroke, ring or leg. In Amharic that small mark carries
grammar, not just pronunciation: **ሰበረ** "he broke", **ሰበሩ** "they broke" and **ስበር** "break!" differ only
in vowel orders. One wrong vowel form is a different word.

Two experiences led me to this gap.

* **As a reader and writer of Amharic.** When I use AI tools in Amharic, the output looks fluent
  but carries wrong vowel forms: the right consonant with the wrong order. A Latin-script reader
  would not notice. An Amharic reader sees broken grammar.
* **As a researcher on Amharic OCR.** My course project proposal (*A Controlled Architecture
  Comparison of CNN-LSTM, Attention, and Transformer Recognizers for the Fidel Amharic OCR Dataset*)
  builds on the Fidel benchmark (Chamisso et al., 2026). Its error analysis found that **most
  substitution errors across every model are the correct consonant with the wrong vowel form**,
  and it proposed no remedy. My proposal attacks this in dedicated OCR models with a *factored*
  output layer that predicts the consonant (33-way) and the vowel order (7-way) separately.

The question here is whether general-purpose frontier models, the ones people actually use, share
the same blind spot. My hypothesis: **they treat each Fidel character as an atomic, opaque symbol.
They know neither the consonant × vowel structure of the script nor how to see it.**

### Why standard benchmarks miss it

* Multilingual benchmarks such as FLORES, Belebele, Global-MMLU and AfriMMLU score *semantic*
  tasks at the sentence level. A model can pick the right multiple-choice answer while being unable
  to spell. The vowel-form structure is never tested directly.
* OCR and document benchmarks such as OCRBench, DocVQA and CC-OCR are dominated by Latin and CJK
  text. Where Ethiopic appears, results are aggregated into a single CER, and **CER does not
  distinguish a vowel-form slip from any other error**.
* Tokenizers hide the problem. Byte-level BPE assigns each Fidel character one or more opaque ids,
  so the model gets no signal that ቡ and ቦ share a consonant.
* There is no natural control. A low Amharic score is usually written off as "low-resource". The
  structural question needs a *matched* comparison against a script with the same
  consonant × vowel composition and plenty of data. Korean Hangul (바 버 보 부 …) is such a script.

---

## 2. Evaluation

### Model

[`Qwen/Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct): a 2B-parameter
vision-language model from a current frontier open-weight family, trending on the Hub. It is
multilingual, does OCR, and is small enough to run on a free Colab T4 or even CPU. Decoding is greedy,
with an answer-only system prompt and an `Answer:` prefill so that the answer can be parsed reliably.

### Design: one structural question, three ways, with a matched control

Every gold answer is computed from Unicode arithmetic (Ethiopic: `0x1200 + 8·row + order`; Hangul:
`0xAC00 + (lead·21 + vowel)·28 + tail`), so scoring needs no human judgment. Every Ethiopic task has a
**Hangul control** with the same template, scheme description and answer format. Hangul is also a
consonant × vowel script, but far better represented in training data. The control separates
"the model has not learned this script's structure" from "syllabic scripts are hard".

| Probe | Items | Question |
|---|---|---|
| **Symbolic**: `decompose_consonant`, `identify_order`, `compose`, `transform` | 4 × (231 Ethiopic + 133 Hangul) | Does the model *know* the grid? The 7-order scheme is explained in every prompt with a worked example, so this tests applying it, not recalling it. |
| **Minimal pairs**: `minimal_pair` | 288 Ethiopic + 246 Hangul | Can it *notice* a single vowel-form change in a real word, compared with a single consonant change? |
| **Perception**: `ocr_char`, `ocr_word`, `ocr_line` | 231 + 96 + 30 Ethiopic; 133 + 82 + 30 Hangul | Can it *see* the difference? Rendered in Noto Sans, so there is no handwriting noise; this is a best-case upper bound. |
| **Tokenizer** | 231 + 133 characters | Is the consonant/vowel structure even visible in the input ids? |

OCR errors are aligned by edit distance. Every substitution is then classified as a **vowel-form error**
(same consonant, wrong order: the error type the Fidel benchmark found dominant), a consonant error,
both, or an **other-script** error (the model wrote Latin, Greek, Georgian or other characters).

### Reproduce

```bash
cd fidelprobe
python build_dataset.py --out ../data            # deterministic, seed 0
python run_eval.py --model Qwen/Qwen3-VL-2B-Instruct --data ../data --out ../results
python tokenizer_probe.py --out ../results
python analyze.py --data ../data --results ../results
```

Or open `fidelprobe/FidelProbe_colab.ipynb` in Colab (T4) and run all.

<!-- RESULTS -->

---

## 3. Path forward

The results point to three separable failures: the model cannot *see* vowel strokes, it does not
*know* the grid, and its tokenizer *hides* the grid. Each needs its own fix, and they compound.

### 3.1 Data curation: teach the grid explicitly, with vowel-form hard negatives

* **Fidel drills generated from Unicode arithmetic.** The grid is fully regular, so unlimited
  verified supervision is free: decompose, compose, transform, transliterate, and "what changed
  between ሰበረ and ሰበሩ?". These become instruction-tuning data, or verifiable rewards for RL
  (every answer is checkable by code, like math). This is the curriculum every Ethiopian
  first-grader goes through (ሀ ሁ ሂ ሃ ሄ ህ ሆ), and models have never been given it.
* **Vowel-form hard negatives for vision.** For every real Amharic word, render or collect its
  vowel-form neighbours: same consonants, one order changed, often another valid word. Train
  image–text contrastive and OCR objectives against them. Generic captioning data never forces a
  model to look at the small stroke that separates ሰ from ሱ, ሲ, ሳ, ሴ, ስ, ሶ.
* **Real handwriting and print.** The Fidel dataset (366k line images from 411 writers, CC-BY)
  plus multi-font rendering: Noto, Abyssinica SIL, Nyala and Power Geez fonts, with degradation.
  Today there is so little Ethiopic in VLM pre-training that the model falls back to
  similar-looking scripts. That is exactly the other-script error measured here.
* **Grammar-aware text data.** Vowel orders carry Amharic morphology (person, number, mood).
  Curate morphologically annotated text, with paradigm tables of verb roots across orders, so that
  the language model learns that a vowel change is a grammar change, not noise.

### 3.2 Architecture: make the consonant × vowel factorisation explicit

* **Factored character embeddings (input side).** Represent each Ethiopic character as
  `E_consonant[row] + E_order[order]` (plus a residual per-character embedding), instead of an
  opaque BPE id or a byte sequence split differently for each form. This is the input-side version
  of the **factored output layer** in my OCR proposal. It shares statistical strength across all 7
  forms of a consonant and makes a vowel change a single, learnable direction in embedding space.
  It also costs ~1 token per character instead of 1.5.
* **Factored output head.** Predict consonant (33-way) and order (7-way) as two heads whose
  log-probabilities are summed. This is the same idea as the Bengali.AI grapheme root / diacritic
  decomposition, applied inside an LLM or VLM decoder for Ethiopic code points. It could be a
  drop-in adapter on the LM head, restricted to the Ethiopic block.
* **Script-aware tokenizer surgery.** At minimum, ensure that all 231 core characters are single
  tokens (today only about half are) and initialise new token embeddings from consonant + order
  means. Byte-level fallback splits ሂ into a byte prefix shared across *different* consonants. That
  is the wrong grouping: the vowel order sits in the low 3 bits of the code point, so a
  structure-aware merge order is possible.
* **Higher-resolution glyph crops for the vision tower.** Vowel strokes are a few pixels wide.
  Dynamic-resolution encoders should up-weight small-glyph scripts, or use a glyph-level auxiliary
  head that predicts (consonant, order) per character region.

### 3.3 Fine-tuning paradigm

1. **Stage 1, structural SFT** (LoRA, cheap): Fidel drills plus vowel-form minimal pairs. Text-only,
   generated, verifiable.
2. **Stage 2, perceptual alignment:** OCR fine-tuning on Fidel plus rendered text, with an auxiliary
   factored (consonant, order) loss and a contrastive loss against vowel-form hard negatives.
3. **Stage 3, RL with verifiable rewards:** reward exact character match, with a larger penalty
   for vowel-form errors that change morphology.
4. **Evaluation:** report the **vowel-form confusion rate** alongside CER for every Ethiopic
   model. Every syllabic script (Ethiopic, Devanagari, Bengali, Kannada, Canadian Syllabics) needs a
   structural error breakdown like this one. A single CER hides the failure that matters most to
   readers.

---

## License and attribution

Code and generated probes: CC-BY-4.0. Model weights: Apache-2.0 (Qwen). Amharic and Korean word and
sentence lists were written for this benchmark. Related data: Fidel dataset (Chamisso et al., 2026),
`upanzi/fidel-dataset`.
