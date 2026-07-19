# Annotation Guide — Chinese PII for OPF Finetuning

The synthetic generator (`gen_cn_finetune_data.py`) covers most of what you need,
but models trained purely on templated data tend to over-fit the surface and
under-perform on real prose. **Mixing 100–300 hand-labeled real sentences into
the training set almost always lifts final precision/recall more than doubling
the synthetic count.** This guide shows the fastest way to do that.

## TL;DR

1. Start the sidecar on the base checkpoint.
2. Run the sidecar against your raw text and dump its predictions.
3. Open each prediction in an editor, correct the offsets, save as JSONL.
4. Append to your training JSONL. Train. Done.

The schema you write matches what `opf train` expects verbatim.

---

## 1. Run the sidecar

```bash
python serve.py \
  --checkpoint ~/.opf/privacy_filter \
  --device auto \
  --port 8799
```

Wait for `/health` to return `model_loaded: true`.

## 2. Pull predictions on a corpus

For each text fragment (one record = one paragraph / one chat message / one
log line):

```bash
curl -s -X POST http://127.0.0.1:8799/redact \
  -H 'Content-Type: application/json' \
  -d '{"texts":["王伟的邮箱是wangwei@example.com，手机13800138000，住在北京市朝阳区建国路88号。"]}'
```

You'll get back the redacted text + a mapping of sentinels → originals.

## 3. Convert predictions to a starter JSONL record

Use `scripts/predictions_to_jsonl.py` (provided) to convert a batch of
`(text, predictions)` tuples into OPF-format JSONL with one record per line:

```bash
python scripts/predictions_to_jsonl.py \
  --input predictions.jsonl \
  --output to_review.jsonl
```

`predictions.jsonl` format expected (one JSON per line):

```json
{"text": "王伟的邮箱是wangwei@example.com，手机13800138000，住在北京市朝阳区建国路88号。", "predicted": [{"label": "private_person", "start": 0, "end": 2}, {"label": "private_email", "start": 7, "end": 25}, {"label": "private_phone", "start": 28, "end": 39}, {"label": "private_address", "start": 42, "end": 53}]}
```

## 4. Fix it up

Open `to_review.jsonl` in your editor of choice. For each line:

- **Offsets are wrong** → adjust `start`/`end` so `text[start:end]` matches the
  intended surface exactly.
- **Span missed** → add a new entry to `spans` (or `label` list).
- **False positive** → remove the entry.
- **Wrong category** → change the label.

A practical review pattern is a 3-line per-record display:

```
TEXT:    王伟的邮箱是wangwei@example.com，手机13800138000，住在北京市朝阳区建国路88号。
SPANS:   private_person 王伟 @0..2       ✓
         private_email  wangwei@example.com @7..25   ✓
         private_phone  13800138000 @28..39   ✓
         private_address 北京市朝阳区建国路88号 @42..53   ✓
```

### Rules of thumb

- **Character offsets, not byte offsets.** 中文 characters each count as 1.
- **The full surface, including prefixes.** `+86 13800138000` is one span, not
  two.
- **No nested spans.** If `private_address` already covers `北京市朝阳区`, do
  not also add `朝阳区` as a separate address span.
- **Punctuation around the surface is usually not part of the span.** `王伟，`
  should be `王伟` only — the trailing `，` stays outside.

## 5. Merge into the training set

```bash
# generate synthetic base
python scripts/gen_cn_finetune_data.py \
  --out data/train_cn.jsonl \
  --n 2000 \
  --val-frac 0.1

# concatenate reviewed real-data records
cat data/train_cn.jsonl data/reviewed_real.jsonl > data/train_combined.jsonl

# (re)split val from train if you haven't done so already
python -c "
import json, random
rows = [json.loads(l) for l in open('data/train_combined.jsonl')]
random.Random(0).shuffle(rows)
val, train = rows[:200], rows[200:]
open('data/train.jsonl', 'w').writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in train)
open('data/val.jsonl',   'w').writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in val)
print('train', len(train), 'val', len(val))
"
```

## 6. Train

```bash
opf train data/train.jsonl \
  --validation-dataset data/val.jsonl \
  --output-dir ./opf_cn_ft \
  --checkpoint ~/.opf/privacy_filter \
  --epochs 2 \
  --batch-size 2 \
  --learning-rate 1e-5
```

## 7. Validate before deploying

```bash
opf eval data/holdout_real.jsonl --checkpoint ./opf_cn_ft
```

A good holdout is **never-touched real Chinese text** — different domain from
your training pool (e.g. if you trained on chat, hold out support tickets).

## What to label, in priority order

1. **CN phone numbers** — the most consistent signal in real text. Capture
   `+86 / (86) / 0XX-XXXX-XXXX / plain 11-digit` variants.
2. **Email with CN domains** — `.com.cn / .cn / qq.com / 163.com` etc. The
   default model has the weakest coverage here.
3. **CN addresses** — full 省市区路号 if available; even partial addresses are
   worth labeling.
4. **CN person names** — most volatile. 2-char vs 3-char (compound surname)
   matters, plus 4-char (compound + 2-char given). When in doubt, include both
   name mentions in the same sentence so the model sees full context.

## Quality bar

| Bad record                                              | Why it's bad                        | Fix                                  |
|---------------------------------------------------------|-------------------------------------|--------------------------------------|
| `text` says `王伟` but `spans` has `start: 0, end: 1`   | Wrong offsets, off-by-one           | Use `text.index(...)` to find it     |
| `label: private_person` but surface is `王伟先生`        | Honorific included in the surface   | Drop `先生` or split into two spans  |
| Same surface labeled twice with conflicting offsets     | Most likely a typo                  | Pick the correct one, delete the rest |
| `private_address` for `北京市` only                     | The address is incomplete           | Extend to `北京市朝阳区建国路88号` if known, otherwise drop |

## When to stop annotating

Stop when:

- `opf eval` on your held-out real-data set shows no further precision/recall
  gain for ~50 added records.
- You're seeing the same false-positive patterns recur (those need lexicon
  additions, not more data).
- Diminishing returns: 1 day of annotation improves eval by <1% — at that point
  you're done and you can move on to a tighter regex prefilter for the
  remaining misses.

## Common pitfalls

- **Don't mix byte and character offsets.** Python `str.index()` is char-based;
  `bytes.index()` is byte-based. OPF uses char.
- **Don't trust tiktoken.** The model's tokenizer may split a Chinese address
  across multiple tokens. The trainer handles this via `token_char_ranges`, so
  as long as your char offsets are correct, the model will see the right labels.
- **Don't duplicate surfaces.** If `王伟` appears twice in a sentence, label
  both occurrences — `spans` accepts a list of `[start, end]` pairs per key.