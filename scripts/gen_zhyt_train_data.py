#!/usr/bin/env python3
"""
Generate a labeled JSONL finetune dataset for OpenAI Privacy Filter,
focused on Traditional Chinese regions: Taiwan (zh-TW), Hong Kong (zh-HK),
and Macau (zh-MO).

This script is the training-side counterpart to gen_zhyt_test_data.py:
the two share lexicons (zhyt_lexicons.py) so a model trained on this
output is evaluated on a held-out test set with consistent phone,
address, and naming conventions across both files.

The output schema matches what `opf train` consumes via
opf/_eval/preprocess.py::parse_record:
  * `text`     (str)
  * `spans`    (dict["label: surface": [[start, end], ...]])  -- preferred
  * `info`     ({id, source, region, ...})                    -- optional

Usage:
    python scripts/gen_zhyt_train_data.py \\
        --out data/zyht_finetune/train.jsonl \\
        --n 2000 \\
        --seed 31
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Callable, Iterable

# Shared lexicons (used by gen_zhyt_test_data.py as well).
from zhyt_lexicons import (
    TW_MOBILE_PREFIXES, TW_LANDLINE_AREA_CODES,
    TW_CITIES_DISTRICTS_ROADS, TW_STREET_NO, TW_FLOOR, TW_UNIT,
    TW_COMPOUND_SURNAMES, TW_SURNAMES, TW_GIVEN_CHARS,
    HK_MOBILE_PREFIXES, HK_LANDLINE_PREFIXES,
    HK_DISTRICTS_STREETS, HK_STREET_NO, HK_FLOOR, HK_UNIT,
    HK_COMPOUND_SURNAMES, HK_SURNAMES, HK_GIVEN_CHARS,
    MO_MOBILE_PREFIXES, MO_LANDLINE_PREFIXES,
    MO_DISTRICTS_STREETS, MO_STREET_NO, MO_FLOOR, MO_UNIT,
    MO_COMPOUND_SURNAMES, MO_SURNAMES, MO_GIVEN_CHARS,
    EMAIL_DOMAINS_TW, EMAIL_DOMAINS_HK_MO, EMAIL_LOCALPARTS,
)

# =============================================================================
# Entity generators
# =============================================================================

# ---- Taiwan ----
def tw_phone(rng: random.Random) -> str:
    kind = rng.choices(["mobile", "landline"], weights=[3, 2])[0]
    if kind == "mobile":
        prefix = rng.choice(TW_MOBILE_PREFIXES)
        body = "".join(str(rng.randint(0, 9)) for _ in range(6))
        style = rng.choices(["dashed", "plain", "spaces"], weights=[4, 2, 1])[0]
        if style == "dashed":
            return f"{prefix}-{body[:3]}-{body[3:]}"
        if style == "spaces":
            return f"{prefix} {body[:3]} {body[3:]}"
        return f"{prefix}{body}"
    else:
        area = rng.choice(TW_LANDLINE_AREA_CODES)
        if len(area) == 2:
            local = "".join(str(rng.randint(0, 9)) for _ in range(8))
            return f"({area}) {local[:4]}-{local[4:]}"
        else:
            local = "".join(str(rng.randint(0, 9)) for _ in range(7))
            return f"({area}) {local[:3]}-{local[3:]}"


def tw_name(rng: random.Random) -> str:
    if rng.random() < 0.10:
        surname = rng.choice(TW_COMPOUND_SURNAMES)
        given_len = rng.choice([1, 2])
    else:
        surname = rng.choice(TW_SURNAMES)
        given_len = rng.choices([1, 2], weights=[3, 7])[0]
    given = "".join(rng.choice(TW_GIVEN_CHARS) for _ in range(given_len))
    return surname + given


def tw_address(rng: random.Random) -> str:
    city, district, road = rng.choice(TW_CITIES_DISTRICTS_ROADS)
    number = rng.choice(TW_STREET_NO)
    floor  = rng.choice(TW_FLOOR)
    unit   = rng.choice(TW_UNIT)
    parts = [city, district, f"{road}{number}號"]
    if floor: parts.append(floor)
    if unit:  parts.append(unit)
    return "".join(parts)


def tw_email(rng: random.Random) -> str:
    return rng.choice(EMAIL_LOCALPARTS) + rng.choice(EMAIL_DOMAINS_TW)


# ---- Hong Kong ----
def hk_phone(rng: random.Random) -> str:
    kind = rng.choices(["mobile", "landline"], weights=[4, 1])[0]
    prefix = (rng.choice(HK_MOBILE_PREFIXES) if kind == "mobile"
              else rng.choice(HK_LANDLINE_PREFIXES))
    body = "".join(str(rng.randint(0, 9)) for _ in range(4))
    style = rng.choices(["dashed", "plain"], weights=[5, 1])[0]
    return f"{prefix}-{body}" if style == "dashed" else f"{prefix}{body}"


def hk_name(rng: random.Random) -> str:
    if rng.random() < 0.08:
        surname = rng.choice(HK_COMPOUND_SURNAMES)
        given_len = rng.choice([1, 2])
    else:
        surname = rng.choice(HK_SURNAMES)
        given_len = rng.choices([1, 2], weights=[3, 7])[0]
    given = "".join(rng.choice(HK_GIVEN_CHARS) for _ in range(given_len))
    return surname + given


def hk_address(rng: random.Random) -> str:
    district, street = rng.choice(HK_DISTRICTS_STREETS)
    number = rng.choice(HK_STREET_NO)
    floor  = rng.choice(HK_FLOOR)
    unit   = rng.choice(HK_UNIT)
    parts = ["香港", district, f"{street}{number}號"]
    if floor: parts.append(floor)
    if unit:  parts.append(unit)
    return "".join(parts)


def hk_email(rng: random.Random) -> str:
    return rng.choice(EMAIL_LOCALPARTS) + rng.choice(EMAIL_DOMAINS_HK_MO)


# ---- Macau ----
def mo_phone(rng: random.Random) -> str:
    kind = rng.choices(["mobile", "landline"], weights=[4, 1])[0]
    prefix = (rng.choice(MO_MOBILE_PREFIXES) if kind == "mobile"
              else rng.choice(MO_LANDLINE_PREFIXES))
    body = "".join(str(rng.randint(0, 9)) for _ in range(4))
    style = rng.choices(["dashed", "plain"], weights=[5, 1])[0]
    return f"{prefix}-{body}" if style == "dashed" else f"{prefix}{body}"


def mo_name(rng: random.Random) -> str:
    if rng.random() < 0.05:
        surname = rng.choice(MO_COMPOUND_SURNAMES)
        given_len = rng.choice([1, 2])
    else:
        surname = rng.choice(MO_SURNAMES)
        given_len = rng.choices([1, 2], weights=[3, 7])[0]
    given = "".join(rng.choice(MO_GIVEN_CHARS) for _ in range(given_len))
    return surname + given


def mo_address(rng: random.Random) -> str:
    district, street = rng.choice(MO_DISTRICTS_STREETS)
    number = rng.choice(MO_STREET_NO)
    floor  = rng.choice(MO_FLOOR)
    unit   = rng.choice(MO_UNIT)
    parts = ["澳門", district, f"{street}{number}號"]
    if floor: parts.append(floor)
    if unit:  parts.append(unit)
    return "".join(parts)


def mo_email(rng: random.Random) -> str:
    return rng.choice(EMAIL_LOCALPARTS) + rng.choice(EMAIL_DOMAINS_HK_MO)


# =============================================================================
# Templates (Traditional Chinese sentences)
#
# Each template returns (sentence, [(label, surface), ...]) where each surface
# appears exactly once in the sentence (build_spans enforces this assumption).
# =============================================================================

# ---- Taiwan ----
def tw_phone_template(rng: random.Random) -> tuple[str, list[tuple[str, str]]]:
    p = tw_phone(rng)
    if rng.random() < 0.75:
        n = tw_name(rng)
        sents = [
            f"請撥打{p}聯絡{n}。",
            f"{n}的手機號碼是{p}，方便時請回電。",
            f"{n}的聯絡電話：{p}（上班時間）。",
            f"{n}的行動電話為{p}，歡迎來電。",
            f"如需聯絡{n}，請撥{p}。",
        ]
        return rng.choice(sents), [("private_person", n), ("private_phone", p)]
    sents = [
        f"緊急聯絡電話：{p}。",
        f"簡訊已發送至{p}，請查收。",
        f"請將驗證碼發送至{p}，謝謝。",
        f"來電號碼 {p}，未接聽。",
        f"訊息已轉送至{p}。",
    ]
    return rng.choice(sents), [("private_phone", p)]


def tw_email_template(rng: random.Random) -> tuple[str, list[tuple[str, str]]]:
    e = tw_email(rng)
    if rng.random() < 0.55:
        n = tw_name(rng)
        sents = [
            f"{n}的電子郵件是{e}，歡迎來信。",
            f"{n}的聯絡郵箱：{e}。",
            f"{n}的 Email：{e}。",
            f"{n}的工作信箱為{e}。",
        ]
        return rng.choice(sents), [("private_person", n), ("private_email", e)]
    sents = [
        f"請將資料寄送至{e}。",
        f"客服郵箱：{e}。",
        f"註冊時使用的電子郵件是{e}。",
        f"技術支援請聯絡{e}。",
        f"回饋信箱：{e}。",
        f"活動報名請寄信至{e}。",
        f"退款申請請發送至{e}。",
    ]
    return rng.choice(sents), [("private_email", e)]


def tw_address_template(rng: random.Random) -> tuple[str, list[tuple[str, str]]]:
    a = tw_address(rng)
    if rng.random() < 0.55:
        n = tw_name(rng)
        sents = [
            f"收件地址：{a}，收件人：{n}。",
            f"{n}的住址是{a}。",
            f"{n}的戶籍地址是{a}。",
            f"{n}目前居住於{a}。",
        ]
        return rng.choice(sents), [("private_person", n), ("private_address", a)]
    sents = [
        f"請將貨物寄往{a}。",
        f"辦公地點位於{a}。",
        f"戶籍地址：{a}。",
        f"收貨地址：{a}。",
        f"出貨地址：{a}。",
        f"目的地：{a}。",
    ]
    return rng.choice(sents), [("private_address", a)]


def tw_combo_template(rng: random.Random) -> tuple[str, list[tuple[str, str]]]:
    n = tw_name(rng)
    p = tw_phone(rng)
    e = tw_email(rng)
    a = tw_address(rng)
    sents = [
        f"{n}（電話{p}，郵箱{e}）現居{a}。",
        f"客戶{n}的聯絡資訊如下：地址{a}，手機{p}，郵箱{e}。",
        f"{n}的檔案：地址{a}，聯絡方式{p}，電子郵件{e}。",
        f"關於{n}：住址{a}、電話{p}、Email：{e}。",
        f"{n}｜地址：{a}｜手機：{p}｜郵箱：{e}。",
    ]
    return rng.choice(sents), [
        ("private_person",  n),
        ("private_phone",   p),
        ("private_email",   e),
        ("private_address", a),
    ]


# ---- Hong Kong ----
def hk_phone_template(rng: random.Random) -> tuple[str, list[tuple[str, str]]]:
    p = hk_phone(rng)
    if rng.random() < 0.75:
        n = hk_name(rng)
        sents = [
            f"請致電{p}聯絡{n}。",
            f"{n}的電話號碼是{p}，方便時請回覆。",
            f"{n}的聯絡電話：{p}（辦公時間）。",
            f"{n}的行動電話為{p}，歡迎來電。",
            f"如需聯絡{n}，請撥{p}。",
        ]
        return rng.choice(sents), [("private_person", n), ("private_phone", p)]
    sents = [
        f"緊急聯絡電話：{p}。",
        f"短訊已發送至{p}，請查收。",
        f"請將驗證碼發送至{p}，謝謝。",
        f"來電號碼 {p}，未接聽。",
        f"訊息已轉送至{p}。",
    ]
    return rng.choice(sents), [("private_phone", p)]


def hk_email_template(rng: random.Random) -> tuple[str, list[tuple[str, str]]]:
    e = hk_email(rng)
    if rng.random() < 0.55:
        n = hk_name(rng)
        sents = [
            f"{n}的電郵地址是{e}，歡迎來信。",
            f"{n}的聯絡電郵：{e}。",
            f"{n}的 Email：{e}。",
            f"{n}的工作電郵為{e}。",
        ]
        return rng.choice(sents), [("private_person", n), ("private_email", e)]
    sents = [
        f"請將資料發送至{e}。",
        f"客戶服務電郵：{e}。",
        f"註冊時使用的電郵地址是{e}。",
        f"技術支援請聯絡{e}。",
        f"意見反饋電郵：{e}。",
        f"退款查詢請發送至{e}。",
        f"活動報名請電郵至{e}。",
    ]
    return rng.choice(sents), [("private_email", e)]


def hk_address_template(rng: random.Random) -> tuple[str, list[tuple[str, str]]]:
    a = hk_address(rng)
    if rng.random() < 0.55:
        n = hk_name(rng)
        sents = [
            f"收件地址：{a}，收件人：{n}。",
            f"{n}的住址是{a}。",
            f"{n}的登記地址是{a}。",
            f"{n}目前居住於{a}。",
        ]
        return rng.choice(sents), [("private_person", n), ("private_address", a)]
    sents = [
        f"請將貨品寄往{a}。",
        f"辦公地點位於{a}。",
        f"登記地址：{a}。",
        f"送貨地址：{a}。",
        f"收貨地址：{a}。",
        f"目的地：{a}。",
    ]
    return rng.choice(sents), [("private_address", a)]


def hk_combo_template(rng: random.Random) -> tuple[str, list[tuple[str, str]]]:
    n = hk_name(rng)
    p = hk_phone(rng)
    e = hk_email(rng)
    a = hk_address(rng)
    sents = [
        f"{n}（電話{p}，電郵{e}）現居{a}。",
        f"客戶{n}的聯絡資料如下：地址{a}，手機{p}，電郵{e}。",
        f"{n}的檔案：地址{a}，聯絡方式{p}，電子郵件{e}。",
        f"關於{n}：住址{a}、電話{p}、Email：{e}。",
        f"{n}｜地址：{a}｜手機：{p}｜電郵：{e}。",
    ]
    return rng.choice(sents), [
        ("private_person",  n),
        ("private_phone",   p),
        ("private_email",   e),
        ("private_address", a),
    ]


# ---- Macau ----
def mo_phone_template(rng: random.Random) -> tuple[str, list[tuple[str, str]]]:
    p = mo_phone(rng)
    if rng.random() < 0.75:
        n = mo_name(rng)
        sents = [
            f"請致電{p}聯絡{n}。",
            f"{n}的電話號碼是{p}，方便時請回覆。",
            f"{n}的聯絡電話：{p}（辦公時間）。",
            f"{n}的行動電話為{p}，歡迎來電。",
            f"如需聯絡{n}，請撥{p}。",
        ]
        return rng.choice(sents), [("private_person", n), ("private_phone", p)]
    sents = [
        f"緊急聯絡電話：{p}。",
        f"短訊已發送至{p}，請查收。",
        f"請將驗證碼發送至{p}，謝謝。",
        f"來電號碼 {p}，未接聽。",
        f"訊息已轉送至{p}。",
    ]
    return rng.choice(sents), [("private_phone", p)]


def mo_email_template(rng: random.Random) -> tuple[str, list[tuple[str, str]]]:
    e = mo_email(rng)
    if rng.random() < 0.55:
        n = mo_name(rng)
        sents = [
            f"{n}的電郵地址是{e}，歡迎來信。",
            f"{n}的聯絡電郵：{e}。",
            f"{n}的 Email：{e}。",
            f"{n}的工作電郵為{e}。",
        ]
        return rng.choice(sents), [("private_person", n), ("private_email", e)]
    sents = [
        f"請將資料發送至{e}。",
        f"客戶服務電郵：{e}。",
        f"註冊時使用的電郵地址是{e}。",
        f"技術支援請聯絡{e}。",
        f"意見反饋電郵：{e}。",
        f"退款查詢請發送至{e}。",
        f"活動報名請電郵至{e}。",
    ]
    return rng.choice(sents), [("private_email", e)]


def mo_address_template(rng: random.Random) -> tuple[str, list[tuple[str, str]]]:
    a = mo_address(rng)
    if rng.random() < 0.55:
        n = mo_name(rng)
        sents = [
            f"收件地址：{a}，收件人：{n}。",
            f"{n}的住址是{a}。",
            f"{n}的登記地址是{a}。",
            f"{n}目前居住於{a}。",
        ]
        return rng.choice(sents), [("private_person", n), ("private_address", a)]
    sents = [
        f"請將貨品寄往{a}。",
        f"辦公地點位於{a}。",
        f"登記地址：{a}。",
        f"送貨地址：{a}。",
        f"收貨地址：{a}。",
        f"目的地：{a}。",
    ]
    return rng.choice(sents), [("private_address", a)]


def mo_combo_template(rng: random.Random) -> tuple[str, list[tuple[str, str]]]:
    n = mo_name(rng)
    p = mo_phone(rng)
    e = mo_email(rng)
    a = mo_address(rng)
    sents = [
        f"{n}（電話{p}，電郵{e}）現居{a}。",
        f"客戶{n}的聯絡資料如下：地址{a}，手機{p}，電郵{e}。",
        f"{n}的檔案：地址{a}，聯絡方式{p}，電子郵件{e}。",
        f"關於{n}：住址{a}、電話{p}、Email：{e}。",
        f"{n}｜地址：{a}｜手機：{p}｜電郵：{e}。",
    ]
    return rng.choice(sents), [
        ("private_person",  n),
        ("private_phone",   p),
        ("private_email",   e),
        ("private_address", a),
    ]


REGION_TEMPLATES: dict[str, list[Callable[[random.Random], tuple[str, list[tuple[str, str]]]]]] = {
    "tw": [tw_phone_template, tw_email_template, tw_address_template, tw_combo_template],
    "hk": [hk_phone_template, hk_email_template, hk_address_template, hk_combo_template],
    "mo": [mo_phone_template, mo_email_template, mo_address_template, mo_combo_template],
}

REGION_LABEL = {"tw": "zh-TW", "hk": "zh-HK", "mo": "zh-MO"}

# =============================================================================
# Span construction (offsets + label-key normalization)
# =============================================================================

def build_spans(sentence: str, entities: list[tuple[str, str]]) -> dict[str, list[list[int]]]:
    """
    Convert [(label, surface), ...] into the OPF `spans` dict with unique
    keys of the form "label: surface" and integer character offsets.

    Tracks a per-surface cursor so a template that mentions the same
    surface twice still locates each occurrence correctly.
    """
    spans: dict[str, list[list[int]]] = {}
    cursors: dict[str, int] = {}
    for label, surface in entities:
        if not surface:
            continue
        start_search = cursors.get(surface, 0)
        idx = sentence.index(surface, start_search)
        spans.setdefault(f"{label}: {surface}", []).append([idx, idx + len(surface)])
        cursors[surface] = idx + len(surface)
    return spans


def validate_spans(sentence: str, spans: dict[str, list[list[int]]]) -> None:
    text_len = len(sentence)
    for key, offsets in spans.items():
        for s, e in offsets:
            assert 0 <= s < e <= text_len, f"span {key} out of range"
            assert sentence[s:e] == key.split(": ", 1)[1], (
                f"span {key} doesn't match surface at {s}:{e}"
            )


# =============================================================================
# Main
# =============================================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSONL path (one record per line).")
    parser.add_argument("--n", type=int, default=2000,
                        help="Total records to generate (split evenly across regions).")
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="Fraction of records split into train.val.jsonl.")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--regions", nargs="+", default=["tw", "hk", "mo"],
                        choices=["tw", "hk", "mo"])
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    regions = args.regions
    per_region = args.n // len(regions)
    remainder = args.n - per_region * len(regions)

    records: list[dict] = []
    for region in regions:
        templates = REGION_TEMPLATES[region]
        count = per_region + (1 if remainder > 0 else 0)
        remainder -= 1
        for i in range(count):
            tmpl = rng.choice(templates)
            sentence, entities = tmpl(rng)
            spans = build_spans(sentence, entities)
            validate_spans(sentence, spans)
            records.append({
                "text":  sentence,
                "spans": spans,
                "info":  {
                    "id":     f"{region}_synth_{i:05d}",
                    "source": "scripts.gen_zhyt_train_data",
                    "region": REGION_LABEL[region],
                },
            })

    # Shuffle so the train/val split is class-balanced (combo rate etc. preserved).
    rng.shuffle(records)
    n_val = int(len(records) * args.val_frac)
    val   = records[:n_val]
    train = records[n_val:]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    val_path = args.out.with_suffix(".val.jsonl")
    # train.jsonl = /foo/bar.jsonl → train.val.jsonl = /foo/bar.val.jsonl
    val_path = args.out.parent / (args.out.stem + ".val" + args.out.suffix)
    with val_path.open("w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    label_counts = Counter()
    region_counts = Counter()
    for r in records:
        for k in r["spans"]:
            label_counts[k.split(": ", 1)[0]] += 1
        region_counts[r["info"]["region"]] += 1

    print(f"wrote {len(train)} train + {len(val)} val to {args.out.parent}/")
    print(f"  per region: {dict(region_counts)}")
    print(f"  per label : {dict(label_counts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
