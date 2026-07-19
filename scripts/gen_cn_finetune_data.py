#!/usr/bin/env python3
"""
Generate a labeled JSONL finetune dataset for OpenAI Privacy Filter,
focused on improving detection of:

  * Chinese email domains  (.com.cn, .cn, .edu.cn, .gov.cn, qq.com, ...)
  * Chinese street addresses (省/市/区/路/号 with and without 室/栋/号)
  * Chinese mobile phone numbers (+86 138..., 138-0013-8000, ...)
  * Chinese person names  (2-char, 3-char compound surnames)

The output schema matches what `opf train` consumes via opf/_eval/preprocess.py::parse_record:
  * `text`     (str)
  * `spans`    (dict["label: surface": [[start, end], ...]])  -- preferred
  * `label`    (list[{category, start, end}])                -- accepted fallback
  * `info`     ({id, source, ...})                            -- optional metadata

Offsets are character-based (Python str indices), which is what the trainer expects.

Usage:
    python scripts/gen_cn_finetune_data.py \
        --out train_cn.jsonl \
        --n 2000 \
        --seed 17
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

# -----------------------------------------------------------------------------
# Lexicons
# -----------------------------------------------------------------------------

# Chinese mobile prefixes (publicly known major operator allocations).
# https://zh.wikipedia.org/wiki/中华人民共和国移动电话网络制式
CN_MOBILE_PREFIXES = [
    "130", "131", "132", "133", "134", "135", "136", "137", "138", "139",
    "145", "147", "149",
    "150", "151", "152", "157", "158", "159",
    "165",
    "166",
    "170", "171", "172", "173", "175", "176", "177", "178",
    "180", "181", "182", "183", "184", "185", "186", "187", "188", "189",
    "191", "198", "199",
]

# Chinese landline area codes. Includes the 4 big-city codes (010/020/021/022),
# 3-digit provincial-capitals, and 4-digit prefecture-level cities.
# https://zh.wikipedia.org/wiki/中华人民共和国长途电话区号
CN_LANDLINE_AREA_CODES = [
    # 直辖市 (2-digit area code, 8-digit local)
    "010", "020", "021", "022", "023", "024",
    # Provincial capitals + major cities (3-digit, 7/8-digit local)
    "025", "027", "028", "029", "0311", "0335", "0371", "0379",
    "0431", "0432", "0451", "0471", "0510", "0512", "0513", "0514", "0515",
    "0516", "0517", "0518", "0519", "0523", "0527", "0551", "0552", "0553",
    "0571", "0572", "0573", "0574", "0575", "0576", "0577", "0578", "0579",
    "0591", "0592", "0595", "0663", "0731", "0734", "0743", "0744", "0745",
    "0746", "0750", "0751", "0754", "0755", "0756", "0757", "0758", "0759",
    "0760", "0762", "0763", "0766", "0768", "0769", "0771", "0772", "0773",
    "0774", "0775", "0776", "0777", "0778", "0779", "0790", "0791", "0792",
    "0793", "0794", "0795", "0796", "0797", "0798", "0799", "0851", "0852",
    "0853", "0854", "0855", "0856", "0857", "0858", "0859", "0870", "0871",
    "0872", "0873", "0874", "0875", "0876", "0877", "0878", "0879", "0883",
    "0886", "0887", "0888", "0891", "0892", "0893", "0894", "0895", "0896",
    "0897", "0898", "0899", "0931", "0932", "0933", "0934", "0935", "0936",
    "0937", "0938", "0941", "0943", "0951", "0952", "0953", "0954", "0955",
    "0970", "0971", "0972", "0973", "0974", "0975", "0976", "0977", "0979",
    "0991", "0993", "0994", "0995", "0996", "0997", "0998", "0999",
]

# Some 4-digit area codes (smaller prefecture-level cities / regions).
CN_LANDLINE_AREA_CODES_4 = [
    "0722", "0724", "0728", "0730", "0735", "0736", "0737", "0738", "0739",
    "0740", "0752", "0753", "0761", "0765", "0770", "0781", "0782", "0783",
    "0784", "0785", "0786", "0787", "0788", "0789", "0810", "0812",
    "0813", "0816", "0817", "0818", "0825", "0826", "0827", "0830", "0831",
    "0832", "0833", "0834", "0835", "0836", "0837", "0838", "0839", "0840",
    "0850",
]

# Compound (复姓) + single-char surnames, with a frequency weighting for realism.
COMPOUND_SURNAMES = [
    "欧阳", "司马", "上官", "诸葛", "东方", "皇甫", "尉迟", "公孙",
    "令狐", "宇文", "长孙", "慕容", "司徒", "司空",
]
SINGLE_SURNAMES = [
    "王", "李", "张", "刘", "陈", "杨", "黄", "赵", "吴", "周",
    "徐", "孙", "马", "朱", "胡", "林", "郭", "何", "高", "罗",
    "郑", "梁", "谢", "宋", "唐", "许", "韩", "冯", "邓", "曹",
    "彭", "曾", "萧", "田", "董", "袁", "潘", "蔡", "蒋", "余",
    "于", "杜", "叶", "程", "魏", "苏", "吕", "丁", "任", "沈",
    "姚", "卢", "姜", "崔", "钟", "谭", "陆", "汪", "范", "金",
    "石", "廖", "贾", "夏", "韦", "付", "方", "白", "邹", "孟",
    "熊", "秦", "邱", "江", "尹", "薛", "闫", "段", "雷", "侯",
    "龙", "史", "陶", "黎", "贺", "顾", "毛", "郝", "龚", "邵",
    "万", "钱", "严", "覃", "武", "戴", "莫", "孔", "向", "汤",
]

# Given-name characters (single-char and double-char). Picking from a frequency-
# weighted pool gives us realistic pairings.
GIVEN_CHARS = list("伟芳娜秀英敏静丽强磊军洋勇艳杰娟涛明超秀兰霞平刚桂英文华建国家俊宇浩然子轩梓萱雨欣梓涵思晨一鸣嘉怡欣怡悦心晨曦若曦语桐")

# Email domains beyond the default OPF coverage. Include both TLD-style and
# Chinese-provider style.
EMAIL_DOMAINS = [
    "@gmail.com", "@yahoo.com", "@outlook.com", "@hotmail.com",
    "@example.com", "@qq.com", "@163.com", "@126.com", "@sina.com",
    "@sohu.com", "@aliyun.com", "@139.com", "@189.cn",
    "@example.cn", "@example.com.cn", "@example.edu.cn",
    "@example.gov.cn", "@example.org.cn", "@example.net.cn",
    "@example.ac.cn",
]

# Email local-parts (username). Avoid `.` at end which makes validation tricky.
EMAIL_LOCALPARTS = [
    "wangwei", "li.na", "zhangsan", "liu.yang", "chen.gao",
    "huang.jie", "amy.chen", "bob.li", "test.user", "no.reply",
    "info", "support", "contact", "admin", "hr", "finance",
    "user01", "alice", "bob", "carol", "dave", "eve", "frank",
    "wei.wang", "fang.zhang", "ming.li", "yan.chen",
    "developer", "engineer", "manager", "founder", "ceo",
]

# Cities for address templates.
PROVINCES_CITIES = [
    ("北京市", "朝阳区", "建国路"),
    ("北京市", "海淀区", "中关村大街"),
    ("上海市", "浦东新区", "世纪大道"),
    ("上海市", "徐汇区", "漕溪北路"),
    ("广州市", "天河区", "珠江新城花城大道"),
    ("深圳市", "南山区", "科技园路"),
    ("杭州市", "西湖区", "文三路"),
    ("成都市", "武侯区", "人民南路"),
    ("武汉市", "江汉区", "建设大道"),
    ("南京市", "鼓楼区", "中山北路"),
    ("西安市", "雁塔区", "高新路"),
    ("重庆市", "渝中区", "解放碑"),
    ("苏州市", "姑苏区", "平江路"),
    ("天津市", "和平区", "南京路"),
    ("长沙市", "岳麓区", "麓山南路"),
]

STREET_NO = [str(n) for n in range(1, 200)]
ROOM_SUFFIX = ["", "1栋", "2栋", "3栋", "A栋", "B栋"]
ROOM_NO = ["", "101室", "202室", "503室", "1502室", "3001室"]

# -----------------------------------------------------------------------------
# Generators (each returns a function (rng) -> surface string)
# -----------------------------------------------------------------------------

def gen_phone(rng: random.Random) -> str:
    """A realistic-looking CN mobile OR landline number with varied formatting."""
    kind = rng.choices(["mobile", "landline"], weights=[3, 2])[0]
    if kind == "mobile":
        prefix = rng.choice(CN_MOBILE_PREFIXES)
        # 11-digit total: prefix(3) + body(8)
        body = "".join(str(rng.randint(0, 9)) for _ in range(8))
        raw = prefix + body
        style = rng.choices(
            ["plain", "spaces", "dashes", "intl", "paren", "compact"],
            weights=[3, 2, 2, 2, 1, 1],
        )[0]
        if style == "plain":
            return raw
        if style == "spaces":
            return f"{prefix} {body[:4]} {body[4:]}"
        if style == "dashes":
            return f"{prefix}-{body[:4]}-{body[4:]}"
        if style == "intl":
            return f"+86 {raw}"
        if style == "paren":
            return f"(+86){raw}"
        return raw  # compact
    else:
        # Landline: 2-digit or 3-digit area codes get 8-digit local numbers;
        # 4-digit area codes get 7-digit local numbers (standard CN convention).
        area = rng.choice(CN_LANDLINE_AREA_CODES)
        local_len = 8
        if rng.random() < 0.30:
            # ~30% of landlines use 4-digit area codes (smaller cities).
            area = rng.choice(CN_LANDLINE_AREA_CODES_4)
            local_len = 7
        local = "".join(str(rng.randint(0, 9)) for _ in range(local_len))
        style = rng.choices(
            ["dashed", "plain", "spaces", "intl", "paren"],
            weights=[5, 2, 1, 1, 1],
        )[0]
        if style == "dashed":
            # Most common: 010-12345678 or 0722-2345678
            return f"{area}-{local}"
        if style == "plain":
            return f"{area}{local}"
        if style == "spaces":
            return f"{area} {local}"
        if style == "intl":
            return f"+86 {area} {local}"
        if style == "paren":
            return f"({area}){local}"
        return f"{area}-{local}"


def gen_email(rng: random.Random) -> str:
    local = rng.choice(EMAIL_LOCALPARTS)
    domain = rng.choice(EMAIL_DOMAINS)
    return local + domain


def gen_person_name(rng: random.Random) -> str:
    """Chinese person name: 2 or 3 characters (compound surname + 1 or 2 given)."""
    if rng.random() < 0.18:  # compound-surname rate
        surname = rng.choice(COMPOUND_SURNAMES)
        given_len = rng.choice([1, 2])
    else:
        surname = rng.choice(SINGLE_SURNAMES)
        given_len = rng.choices([1, 2], weights=[3, 7])[0]
    given = "".join(rng.choice(GIVEN_CHARS) for _ in range(given_len))
    return surname + given


def gen_address(rng: random.Random) -> str:
    """Chinese street address: 省/直辖市 + 区 + 路 + 号 (+ 室/栋 optional)."""
    city, district, street = rng.choice(PROVINCES_CITIES)
    number = rng.choice(STREET_NO)
    suffix = rng.choice(ROOM_SUFFIX)
    room = rng.choice(ROOM_NO)
    # Reasonable composition: most addresses omit the optional suffix.
    parts = [city, district, f"{street}{number}号"]
    if suffix:
        parts.append(suffix)
    if room:
        parts.append(room)
    return "".join(parts)


GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "private_phone":   gen_phone,
    "private_email":   gen_email,
    "private_person":  gen_person_name,
    "private_address": gen_address,
}

# -----------------------------------------------------------------------------
# Sentence templates — each picks 1-3 entities and weaves them into Chinese
# prose. We compute character offsets with text.index(surface), which is exact
# only if the surface is unique in the sentence. Templates are designed to make
# every surface unique within the sentence.
# -----------------------------------------------------------------------------

# Each template is a function: rng -> [(label, surface), ...] tuples.
# We then plug them into the sentence in a deterministic order.
# A cleaner template format: sentence can contain {placeholders}; we substitute
# then locate them by index. To keep things simple, we use surface-string
# templates where the surface is already in the sentence and we locate it
# via str.index.
def _phone_template(rng: random.Random) -> tuple[str, list[tuple[str, str]]]:
    p = gen_phone(rng)
    include_name = rng.random() < 0.75
    n = gen_person_name(rng) if include_name else None
    sents_with_n = [
        f"请拨打{p}联系{n}。",
        f"{n}的手机号是{p}，方便时请回电。",
        f"{n}的联系电话：{p}（工作时间）。",
    ]
    sents_no_n = [
        f"紧急联系电话：{p}。",
        f"短信已发送至{p}，请查收。",
        f"请将验证码发送至{p}，谢谢。",
        f"来电号码 {p}，未接听。",
    ]
    if include_name:
        s = rng.choice(sents_with_n)
        return s, [("private_person", n), ("private_phone", p)]
    s = rng.choice(sents_no_n)
    return s, [("private_phone", p)]

def _email_template(rng: random.Random) -> tuple[str, list[tuple[str, str]]]:
    e = gen_email(rng)
    include_name = rng.random() < 0.55
    n = gen_person_name(rng) if include_name else None
    sents_with_n = [
        f"{n}的邮箱是{e}，欢迎来信。",
        f"{n}的电子邮箱：{e}。",
    ]
    sents_no_n = [
        f"请将资料发送至{e}。",
        f"我们的客服邮箱：{e}。",
        f"注册时使用的邮箱是{e}。",
        f"技术支持请联系{e}。",
        f"反馈邮箱：{e}。",
    ]
    if include_name:
        s = rng.choice(sents_with_n)
        return s, [("private_person", n), ("private_email", e)]
    s = rng.choice(sents_no_n)
    return s, [("private_email", e)]

def _address_template(rng: random.Random) -> tuple[str, list[tuple[str, str]]]:
    a = gen_address(rng)
    include_name = rng.random() < 0.55
    n = gen_person_name(rng) if include_name else None
    sents_with_n = [
        f"收件地址：{a}，收件人：{n}。",
        f"{n}的住址是{a}。",
        f"{n}的户籍地址是{a}。",
    ]
    sents_no_n = [
        f"请将货物寄往{a}。",
        f"办公地点位于{a}。",
        f"户籍地址：{a}。",
        f"收货地址：{a}。",
        f"出差住址：{a}。",
    ]
    if include_name:
        s = rng.choice(sents_with_n)
        return s, [("private_person", n), ("private_address", a)]
    s = rng.choice(sents_no_n)
    return s, [("private_address", a)]

def _combo_template(rng: random.Random) -> tuple[str, list[tuple[str, str]]]:
    """A combo sentence with phone + email + address + name — for hard cases."""
    n  = gen_person_name(rng)
    p  = gen_phone(rng)
    e  = gen_email(rng)
    a  = gen_address(rng)
    sents = [
        f"{n}（电话{p}，邮箱{e}）现居{a}。",
        f"客户{n}的联系信息如下：地址{a}，手机{p}，邮箱{e}。",
        f"{n}的档案：地址{a}，联系方式{p}，电子邮箱{e}。",
    ]
    s = rng.choice(sents)
    return s, [
        ("private_person",  n),
        ("private_phone",   p),
        ("private_email",   e),
        ("private_address", a),
    ]

ALL_TEMPLATES = [_phone_template, _email_template, _address_template, _combo_template]

# -----------------------------------------------------------------------------
# Span construction (offsets + label-key normalization)
# -----------------------------------------------------------------------------

def build_spans(sentence: str, entities: list[tuple[str, str]]) -> dict[str, list[list[int]]]:
    """
    Convert [(label, surface), ...] into the OPF `spans` dict with unique
    keys of the form "label: surface" and integer character offsets.

    Strategy: track a per-surface cursor so the same surface can appear more
    than once in a sentence (e.g. a name mentioned twice in a combo template)
    and we still locate each occurrence correctly.
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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSONL path (one record per line).")
    parser.add_argument("--n", type=int, default=2000,
                        help="Number of examples to generate.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="Fraction written to <out-stem>.val.jsonl")
    parser.add_argument("--write-label-style", action="store_true",
                        help="Also emit the alternate 'label' field, not just 'spans'.")
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    records: list[dict] = []

    for i in range(args.n):
        tmpl = rng.choice(ALL_TEMPLATES)
        sentence, entities = tmpl(rng)
        spans = build_spans(sentence, entities)
        validate_spans(sentence, spans)

        rec: dict = {
            "text":  sentence,
            "spans": spans,
            "info":  {
                "id":     f"cn_synth_{i:05d}",
                "source": "scripts.gen_cn_finetune_data",
            },
        }
        if args.write_label_style:
            # Mirror spans into the alternate `label` schema for compatibility.
            label_list: list[dict] = []
            for key, offsets in spans.items():
                cat = key.split(": ", 1)[0]
                for s, e in offsets:
                    label_list.append({"category": cat, "start": s, "end": e})
            rec["label"] = label_list

        records.append(rec)

    # Split train / val (deterministic via shuffle-seed)
    rng.shuffle(records)
    n_val = max(1, int(args.n * args.val_frac))
    val, train = records[:n_val], records[n_val:]

    out_train = args.out
    out_val = args.out.with_suffix("") if args.out.suffix == ".jsonl" else args.out
    out_val = Path(str(out_val) + ".val.jsonl")

    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_val.parent.mkdir(parents=True, exist_ok=True)

    for path, rows in [(out_train, train), (out_val, val)]:
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows):>5} records to {path}")

    # Sanity-check: print one example from each file.
    print("\n--- sample train record ---")
    print(json.dumps(train[0], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())