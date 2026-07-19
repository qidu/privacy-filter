#!/usr/bin/env python3
"""
Generate a labeled JSONL TEST dataset for OpenAI Privacy Filter, covering
Traditional Chinese regions:

  * Taiwan (zh-TW): 台北/台中/高雄..., 09XX-XXX-XXX mobile, (0X) XXXX-XXXX landline
  * Hong Kong (zh-HK): 香港 + 區 + 街, 9XXX-XXXX mobile, 2/3XXX-XXXX landline
  * Macau (zh-MO): 澳門 + 區, 6XXX-XXXX mobile, 28XX-XXXX landline

This script is intentionally separate from the main CN training-data generator
(`gen_cn_finetune_data.py`) because:

  1. We want a held-out test set that the model never sees during training,
     to measure generalization across Chinese-script variants.
  2. Phone/address/naming conventions differ enough that mixing the templates
     makes both the data and the script harder to audit.

The output schema matches `opf train` / `opf eval` (see parse_record):
  text + spans + info.

Usage:
    python scripts/gen_zhyt_test_data.py \\
        --out data/zhyt_eval/test.jsonl \\
        --n 1000 \\
        --seed 31
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Callable, Iterable

# =============================================================================
# Taiwan (zh-TW) — Traditional Chinese, Taiwan-specific conventions
# =============================================================================

TW_MOBILE_PREFIXES = [
    "0910", "0911", "0912", "0913", "0914", "0915", "0916", "0917", "0918", "0919",
    "0920", "0921", "0922", "0923", "0924", "0925", "0926", "0927", "0928", "0929",
    "0930", "0931", "0932", "0933", "0934", "0935", "0936", "0937", "0938", "0939",
    "0952", "0953", "0954", "0955", "0956", "0957", "0958",
    "0960", "0961", "0962", "0963", "0965", "0966", "0967",
    "0970", "0972", "0975", "0976", "0977", "0978", "0979",
    "0980", "0981", "0982", "0983", "0985", "0986", "0987", "0988", "0989",
]

TW_LANDLINE_AREA_CODES = [
    "02", "03", "037", "049",
    "04", "047", "048", "05", "06", "07", "08", "082", "0836", "089", "0826",
]

TW_CITIES_DISTRICTS_ROADS = [
    ("台北市", "大安區", "復興南路一段"),
    ("台北市", "信義區", "松仁路"),
    ("台北市", "中山區", "民生東路二段"),
    ("台北市", "松山區", "八德路三段"),
    ("台北市", "內湖區", "成功路二段"),
    ("台北市", "士林區", "中山北路五段"),
    ("台北市", "文山區", "木柵路二段"),
    ("台北市", "萬華區", "中華路一段"),
    ("新北市", "板橋區", "中山路一段"),
    ("新北市", "永和區", "中山路一段"),
    ("新北市", "中和區", "景平路"),
    ("新北市", "新莊區", "中正路"),
    ("新北市", "三重區", "重新路"),
    ("新北市", "土城區", "中央路"),
    ("桃園市", "桃園區", "中正路"),
    ("桃園市", "中壢區", "中央西路"),
    ("桃園市", "蘆竹區", "南崁路"),
    ("新竹市", "東區", "光復路一段"),
    ("新竹縣", "竹北市", "光明六路"),
    ("台中市", "西區", "美村路一段"),
    ("台中市", "北區", "健行路"),
    ("台中市", "南屯區", "黎明路"),
    ("台中市", "西屯區", "台灣大道三段"),
    ("台中市", "北屯區", "崇德路"),
    ("台南市", "中西區", "海安路二段"),
    ("台南市", "東區", "崇德路"),
    ("台南市", "北區", "成功路"),
    ("台南市", "永康區", "中正路"),
    ("高雄市", "前金區", "中正四路"),
    ("高雄市", "苓雅區", "中正一路"),
    ("高雄市", "三民區", "建國一路"),
    ("高雄市", "左營區", "博愛二路"),
    ("高雄市", "鼓山區", "美術館路"),
    ("基隆市", "中正區", "中正路"),
    ("宜蘭縣", "宜蘭市", "中山路"),
]

TW_STREET_NO = [str(n) for n in range(1, 350)]
TW_FLOOR = ["", "2樓", "3樓", "5樓", "7樓", "8樓", "12樓", "15樓", "20樓"]
TW_UNIT  = ["", "A室", "B室", "C室", "D室", "E室", "F室", "G室"]

TW_COMPOUND_SURNAMES = ["歐陽", "司馬", "上官", "諸葛", "東方", "皇甫", "尉遲", "公孫"]
TW_SURNAMES = [
    "陳", "林", "黃", "張", "李", "王", "吳", "劉", "蔡", "楊",
    "許", "鄭", "謝", "洪", "郭", "邱", "曾", "廖", "賴", "徐",
    "周", "葉", "蘇", "莊", "江", "何", "蕭", "羅", "高", "簡",
    "朱", "鍾", "施", "游", "彭", "藍", "魏", "溫", "涂", "田",
    "鄧", "杜", "侯", "薛", "丁", "傅", "顏", "柯", "白", "連",
    "姚", "邵", "程", "石", "龔", "韓", "阮", "毛", "童", "谷",
]
TW_GIVEN_CHARS = list("家豪志偉俊宏雅婷淑芬怡君冠宇俊傑承翰雅雯欣怡雅琪家瑋怡婷宜君冠廷郁雯怡萱欣儀家銘")

# =============================================================================
# Hong Kong (zh-HK)
# =============================================================================

HK_MOBILE_PREFIXES = [
    "9100", "9111", "9123", "9134", "9145", "9156", "9167", "9178", "9189",
    "9200", "9211", "9222", "9233", "9244", "9255", "9266", "9277", "9288",
    "9300", "9311", "9322", "9333", "9344", "9355", "9366", "9377",
    "9400", "9411", "9422", "9433", "9444", "9455", "9466", "9477", "9488",
    "9500", "9511", "9522", "9533", "9544", "9555", "9566", "9577", "9588",
    "9600", "9611", "9622", "9633", "9644", "9655", "9666", "9677", "9688",
    "9700", "9711", "9722", "9733", "9744", "9755", "9766", "9777", "9788",
    "9800", "9811", "9822", "9833", "9844", "9855", "9866", "9877", "9888", "9899",
]

HK_LANDLINE_PREFIXES = [
    "2123", "2345", "2521", "2567", "2735", "2770", "2832", "2899",
    "3111", "3221", "3456", "3567", "3678", "3789",
]

HK_DISTRICTS_STREETS = [
    ("中環",   "皇后大道中"),
    ("中環",   "德輔道中"),
    ("金鐘",   "金鐘道"),
    ("灣仔",   "軒尼詩道"),
    ("灣仔",   "駱克道"),
    ("銅鑼灣", "軒尼詩道"),
    ("銅鑼灣", "怡和街"),
    ("天后",   "英皇道"),
    ("北角",   "英皇道"),
    ("鰂魚涌", "英皇道"),
    ("太古",   "英皇道"),
    ("上環",   "德輔道西"),
    ("西環",   "德輔道西"),
    ("尖沙咀", "彌敦道"),
    ("尖沙咀", "廣東道"),
    ("尖沙咀", "北京道"),
    ("佐敦",   "彌敦道"),
    ("油麻地", "彌敦道"),
    ("旺角",   "彌敦道"),
    ("旺角",   "亞皆老街"),
    ("太子",   "彌敦道"),
    ("深水埗", "長沙灣道"),
    ("深水埗", "桂林街"),
    ("長沙灣", "長沙灣道"),
    ("荃灣",   "大河道"),
    ("葵涌",   "葵涌道"),
    ("屯門",   "屯門鄉事會路"),
    ("元朗",   "青山公路"),
    ("沙田",   "沙田正街"),
    ("大圍",   "大圍道"),
    ("九龍塘", "窩打老道"),
    ("紅磡",   "漆咸道北"),
    ("土瓜灣", "馬頭圍道"),
    ("黃大仙", "龍翔道"),
    ("鑽石山", "龍翔道"),
]
HK_STREET_NO = [str(n) for n in range(1, 999)]
HK_FLOOR = ["", "2樓", "3樓", "5樓", "8樓", "10樓", "15樓", "20樓", "25樓"]
HK_UNIT  = ["", "A室", "B室", "C室", "D室"]

HK_COMPOUND_SURNAMES = ["歐陽", "司馬", "上官", "諸葛", "司徒", "皇甫"]
HK_SURNAMES = [
    "陳", "林", "黃", "張", "李", "王", "吳", "劉", "蔡", "楊",
    "許", "鄭", "謝", "洪", "郭", "邱", "曾", "廖", "賴", "徐",
    "周", "葉", "蘇", "莊", "江", "何", "蕭", "羅", "高", "簡",
    "朱", "鍾", "施", "游", "彭", "藍", "魏", "溫", "涂", "田",
    "鄧", "杜", "侯", "薛", "丁", "傅", "顏", "柯", "白", "連",
    "姚", "邵", "程", "石", "龔", "韓", "阮", "毛", "童", "谷",
    "甄", "鄺", "黎", "戴", "莫", "姜", "崔", "霍",
]
HK_GIVEN_CHARS = list("家豪志偉俊宏雅婷淑芬怡君冠宇俊傑承翰雅雯欣怡雅琪家瑋怡婷宜君冠廷郁雯怡萱欣儀家銘嘉文曉楓曉怡浩然梓軒梓晴俊賢嘉穎曉彤凱晴")

# =============================================================================
# Macau (zh-MO)
# =============================================================================

MO_MOBILE_PREFIXES = [
    "6200", "6211", "6222", "6233", "6244", "6255", "6266", "6277", "6288", "6299",
    "6300", "6311", "6322", "6333", "6344", "6355", "6366", "6377", "6388", "6399",
    "6500", "6511", "6522", "6533", "6544", "6555", "6566", "6577", "6588", "6599",
    "6600", "6611", "6622", "6633", "6644", "6655", "6666", "6677", "6688", "6699",
    "6800", "6811", "6822", "6833", "6844", "6855", "6866", "6877", "6888", "6899",
]

MO_LANDLINE_PREFIXES = [
    "2830", "2831", "2832", "2833", "2835", "2836", "2837", "2838",
    "2840", "2841", "2842", "2843", "2845", "2846", "2847", "2848",
    "2850", "2851", "2852", "2853", "2855", "2856", "2857", "2858",
    "2860", "2861", "2862", "2863", "2865", "2866", "2867", "2868",
    "2870", "2871", "2872", "2873", "2875", "2876", "2877", "2878",
    "2880", "2881", "2882", "2883", "2885", "2886", "2887", "2888",
    "2890", "2891", "2892", "2893", "2895", "2896", "2897", "2898",
]

MO_DISTRICTS_STREETS = [
    ("中區",         "新馬路"),
    ("中區",         "議事亭前地"),
    ("中區",         "南灣大馬路"),
    ("中區",         "殷皇子大馬路"),
    ("花地瑪堂區",   "罅些喇提督大馬路"),
    ("花地瑪堂區",   "筷子基北街"),
    ("花地瑪堂區",   "祐漢新村第二街"),
    ("花地瑪堂區",   "黑沙環新街"),
    ("花地瑪堂區",   "馬場東大馬路"),
    ("聖安多尼堂區", "高士德大馬路"),
    ("聖安多尼堂區", "美副將大馬路"),
    ("聖安多尼堂區", "連勝街"),
    ("大堂區",       "南灣大馬路"),
    ("大堂區",       "殷皇子大馬路"),
    ("望德堂區",     "美副將大馬路"),
    ("望德堂區",     "荷蘭園大馬路"),
    ("嘉模堂區",     "官也街"),
    ("嘉模堂區",     "地堡街"),
    ("嘉模堂區",     "海洋花園大馬路"),
    ("嘉模堂區",     "華寶花園大馬路"),
    ("路氹金光大道", "路氹連貫公路"),
    ("聖方濟各堂區", "路環連貫公路"),
    ("聖方濟各堂區", "路環石排灣馬路"),
    ("聖方濟各堂區", "竹灣馬路"),
]
MO_STREET_NO = [str(n) for n in range(1, 500)]
MO_FLOOR = ["", "2樓", "3樓", "5樓", "8樓", "10樓", "12樓", "15樓"]
MO_UNIT  = ["", "A座", "B座", "C座", "D座"]

MO_COMPOUND_SURNAMES = ["歐陽", "司馬", "上官", "諸葛", "司徒"]
MO_SURNAMES = [
    "陳", "林", "黃", "張", "李", "王", "吳", "劉", "蔡", "楊",
    "許", "鄭", "謝", "洪", "郭", "邱", "曾", "廖", "賴", "徐",
    "周", "葉", "蘇", "莊", "江", "何", "蕭", "羅", "高", "簡",
    "朱", "鍾", "施", "游", "彭", "藍", "魏", "溫", "涂", "田",
    "鄧", "杜", "侯", "薛", "丁", "傅", "顏", "柯", "白", "連",
    "姚", "邵", "程", "石", "龔", "韓", "阮", "毛", "童", "谷",
    "甄", "鄺", "黎", "戴", "莫", "姜", "崔", "霍", "甘", "陶",
]
MO_GIVEN_CHARS = list("家豪志偉俊宏雅婷淑芬怡君冠宇俊傑承翰雅雯欣怡雅琪家瑋怡婷宜君冠廷郁雯怡萱欣儀家銘嘉文曉楓曉怡浩然梓軒梓晴俊賢嘉穎曉彤凱晴詠儀嘉欣曉雯曉晴詠詩曉琳")

# =============================================================================
# Email domains
# =============================================================================

EMAIL_DOMAINS_TW = [
    "@gmail.com", "@yahoo.com.tw", "@outlook.com", "@hotmail.com",
    "@example.com", "@example.com.tw", "@example.edu.tw", "@example.gov.tw",
]
EMAIL_DOMAINS_HK_MO = [
    "@gmail.com", "@yahoo.com.hk", "@outlook.com", "@hotmail.com",
    "@example.com", "@example.com.hk", "@example.edu.hk", "@example.gov.hk",
    "@example.com.mo", "@example.edu.mo",
]
EMAIL_LOCALPARTS = [
    "ming.wang", "yu.chen", "hsiao.li", "wei.huang", "chia.lin",
    "siu.ling", "ka.fai", "wai.san", "iou.meng", "kin.cheng",
    "info", "support", "contact", "admin", "hr", "finance",
    "user01", "alice", "bob", "carol", "dave", "eve", "frank",
    "developer", "engineer", "manager", "founder", "ceo",
]

# =============================================================================
# Generators
# =============================================================================

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
        else:  # 3 or 4 digit area code
            local = "".join(str(rng.randint(0, 9)) for _ in range(7))
            return f"({area}) {local[:3]}-{local[3:]}"


def hk_phone(rng: random.Random) -> str:
    kind = rng.choices(["mobile", "landline"], weights=[4, 1])[0]
    prefix = (rng.choice(HK_MOBILE_PREFIXES) if kind == "mobile"
              else rng.choice(HK_LANDLINE_PREFIXES))
    body = "".join(str(rng.randint(0, 9)) for _ in range(4))
    style = rng.choices(["dashed", "plain"], weights=[5, 1])[0]
    return f"{prefix}-{body}" if style == "dashed" else f"{prefix}{body}"


def mo_phone(rng: random.Random) -> str:
    kind = rng.choices(["mobile", "landline"], weights=[4, 1])[0]
    prefix = (rng.choice(MO_MOBILE_PREFIXES) if kind == "mobile"
              else rng.choice(MO_LANDLINE_PREFIXES))
    body = "".join(str(rng.randint(0, 9)) for _ in range(4))
    style = rng.choices(["dashed", "plain"], weights=[5, 1])[0]
    return f"{prefix}-{body}" if style == "dashed" else f"{prefix}{body}"


def tw_name(rng: random.Random) -> str:
    if rng.random() < 0.10:
        surname = rng.choice(TW_COMPOUND_SURNAMES)
        given_len = rng.choice([1, 2])
    else:
        surname = rng.choice(TW_SURNAMES)
        given_len = rng.choices([1, 2], weights=[3, 7])[0]
    given = "".join(rng.choice(TW_GIVEN_CHARS) for _ in range(given_len))
    return surname + given


def hk_name(rng: random.Random) -> str:
    if rng.random() < 0.08:
        surname = rng.choice(HK_COMPOUND_SURNAMES)
        given_len = rng.choice([1, 2])
    else:
        surname = rng.choice(HK_SURNAMES)
        given_len = rng.choices([1, 2], weights=[3, 7])[0]
    given = "".join(rng.choice(HK_GIVEN_CHARS) for _ in range(given_len))
    return surname + given


def mo_name(rng: random.Random) -> str:
    if rng.random() < 0.05:
        surname = rng.choice(MO_COMPOUND_SURNAMES)
        given_len = rng.choice([1, 2])
    else:
        surname = rng.choice(MO_SURNAMES)
        given_len = rng.choices([1, 2], weights=[3, 7])[0]
    given = "".join(rng.choice(MO_GIVEN_CHARS) for _ in range(given_len))
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


def hk_address(rng: random.Random) -> str:
    district, street = rng.choice(HK_DISTRICTS_STREETS)
    number = rng.choice(HK_STREET_NO)
    floor  = rng.choice(HK_FLOOR)
    unit   = rng.choice(HK_UNIT)
    parts = ["香港", district, f"{street}{number}號"]
    if floor: parts.append(floor)
    if unit:  parts.append(unit)
    return "".join(parts)


def mo_address(rng: random.Random) -> str:
    district, street = rng.choice(MO_DISTRICTS_STREETS)
    number = rng.choice(MO_STREET_NO)
    floor  = rng.choice(MO_FLOOR)
    unit   = rng.choice(MO_UNIT)
    parts = ["澳門", district, f"{street}{number}號"]
    if floor: parts.append(floor)
    if unit:  parts.append(unit)
    return "".join(parts)


def tw_email(rng: random.Random) -> str:
    return rng.choice(EMAIL_LOCALPARTS) + rng.choice(EMAIL_DOMAINS_TW)


def hk_mo_email(rng: random.Random) -> str:
    return rng.choice(EMAIL_LOCALPARTS) + rng.choice(EMAIL_DOMAINS_HK_MO)

# =============================================================================
# Templates (Traditional Chinese sentences)
# =============================================================================

def _combo(rng, n, p, e, a, *, dialect="tw") -> tuple[str, list]:
    if dialect == "tw":
        sents = [
            f"{n}（電話{p}，郵箱{e}）現居{a}。",
            f"客戶{n}的聯絡資訊如下：地址{a}，手機{p}，郵箱{e}。",
            f"{n}的檔案：地址{a}，聯絡方式{p}，電子郵件{e}。",
        ]
    elif dialect == "hk":
        sents = [
            f"{n}（電話{p}，電郵{e}）現居{a}。",
            f"客戶{n}的聯絡資料如下：地址{a}，手機{p}，電郵{e}。",
            f"{n}的檔案：地址{a}，聯絡方式{p}，電子郵件{e}。",
        ]
    else:  # mo
        sents = [
            f"{n}（電話{p}，電郵{e}）現居{a}。",
            f"客戶{n}的聯絡資料如下：地址{a}，手機{p}，電郵{e}。",
            f"{n}的檔案：地址{a}，聯絡方式{p}，電子郵件{e}。",
        ]
    s = rng.choice(sents)
    return s, [
        ("private_person",  n),
        ("private_phone",   p),
        ("private_email",   e),
        ("private_address", a),
    ]


# ---- Taiwan ----
def tw_phone_template(rng: random.Random) -> tuple[str, list[tuple[str, str]]]:
    p = tw_phone(rng)
    if rng.random() < 0.75:
        n = tw_name(rng)
        sents = [
            f"請撥打{p}聯絡{n}。",
            f"{n}的手機號碼是{p}，方便時請回電。",
            f"{n}的聯絡電話：{p}（上班時間）。",
        ]
        return rng.choice(sents), [("private_person", n), ("private_phone", p)]
    sents = [
        f"緊急聯絡電話：{p}。",
        f"簡訊已發送至{p}，請查收。",
        f"請將驗證碼發送至{p}，謝謝。",
        f"來電號碼 {p}，未接聽。",
    ]
    return rng.choice(sents), [("private_phone", p)]


def tw_email_template(rng: random.Random) -> tuple[str, list[tuple[str, str]]]:
    e = tw_email(rng)
    if rng.random() < 0.55:
        n = tw_name(rng)
        sents = [
            f"{n}的電子郵件是{e}，歡迎來信。",
            f"{n}的聯絡郵箱：{e}。",
        ]
        return rng.choice(sents), [("private_person", n), ("private_email", e)]
    sents = [
        f"請將資料寄送至{e}。",
        f"客服郵箱：{e}。",
        f"註冊時使用的電子郵件是{e}。",
        f"技術支援請聯絡{e}。",
        f"回饋信箱：{e}。",
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
        ]
        return rng.choice(sents), [("private_person", n), ("private_address", a)]
    sents = [
        f"請將貨物寄往{a}。",
        f"辦公地點位於{a}。",
        f"戶籍地址：{a}。",
        f"收貨地址：{a}。",
    ]
    return rng.choice(sents), [("private_address", a)]


def tw_combo_template(rng: random.Random) -> tuple[str, list[tuple[str, str]]]:
    return _combo(rng, tw_name(rng), tw_phone(rng), tw_email(rng),
                  tw_address(rng), dialect="tw")


# ---- Hong Kong ----
def hk_phone_template(rng: random.Random) -> tuple[str, list[tuple[str, str]]]:
    p = hk_phone(rng)
    if rng.random() < 0.75:
        n = hk_name(rng)
        sents = [
            f"請致電{p}聯絡{n}。",
            f"{n}的電話號碼是{p}，方便時請回覆。",
            f"{n}的聯絡電話：{p}（辦公時間）。",
        ]
        return rng.choice(sents), [("private_person", n), ("private_phone", p)]
    sents = [
        f"緊急聯絡電話：{p}。",
        f"短訊已發送至{p}，請查收。",
        f"請將驗證碼發送至{p}，謝謝。",
        f"來電號碼 {p}，未接聽。",
    ]
    return rng.choice(sents), [("private_phone", p)]


def hk_email_template(rng: random.Random) -> tuple[str, list[tuple[str, str]]]:
    e = hk_mo_email(rng)
    if rng.random() < 0.55:
        n = hk_name(rng)
        sents = [
            f"{n}的電郵地址是{e}，歡迎來信。",
            f"{n}的聯絡電郵：{e}。",
        ]
        return rng.choice(sents), [("private_person", n), ("private_email", e)]
    sents = [
        f"請將資料發送至{e}。",
        f"客戶服務電郵：{e}。",
        f"註冊時使用的電郵地址是{e}。",
        f"技術支援請聯絡{e}。",
        f"意見反饋電郵：{e}。",
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
        ]
        return rng.choice(sents), [("private_person", n), ("private_address", a)]
    sents = [
        f"請將貨品寄往{a}。",
        f"辦公地點位於{a}。",
        f"登記地址：{a}。",
        f"送貨地址：{a}。",
    ]
    return rng.choice(sents), [("private_address", a)]


def hk_combo_template(rng: random.Random) -> tuple[str, list[tuple[str, str]]]:
    return _combo(rng, hk_name(rng), hk_phone(rng), hk_mo_email(rng),
                  hk_address(rng), dialect="hk")


# ---- Macau ----
def mo_phone_template(rng: random.Random) -> tuple[str, list[tuple[str, str]]]:
    p = mo_phone(rng)
    if rng.random() < 0.75:
        n = mo_name(rng)
        sents = [
            f"請致電{p}聯絡{n}。",
            f"{n}的電話號碼是{p}，方便時請回覆。",
            f"{n}的聯絡電話：{p}（辦公時間）。",
        ]
        return rng.choice(sents), [("private_person", n), ("private_phone", p)]
    sents = [
        f"緊急聯絡電話：{p}。",
        f"短訊已發送至{p}，請查收。",
        f"請將驗證碼發送至{p}，謝謝。",
        f"來電號碼 {p}，未接聽。",
    ]
    return rng.choice(sents), [("private_phone", p)]


def mo_email_template(rng: random.Random) -> tuple[str, list[tuple[str, str]]]:
    e = hk_mo_email(rng)
    if rng.random() < 0.55:
        n = mo_name(rng)
        sents = [
            f"{n}的電郵地址是{e}，歡迎來信。",
            f"{n}的聯絡電郵：{e}。",
        ]
        return rng.choice(sents), [("private_person", n), ("private_email", e)]
    sents = [
        f"請將資料發送至{e}。",
        f"客戶服務電郵：{e}。",
        f"註冊時使用的電郵地址是{e}。",
        f"技術支援請聯絡{e}。",
        f"意見反饋電郵：{e}。",
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
        ]
        return rng.choice(sents), [("private_person", n), ("private_address", a)]
    sents = [
        f"請將貨品寄往{a}。",
        f"辦公地點位於{a}。",
        f"登記地址：{a}。",
        f"送貨地址：{a}。",
    ]
    return rng.choice(sents), [("private_address", a)]


def mo_combo_template(rng: random.Random) -> tuple[str, list[tuple[str, str]]]:
    return _combo(rng, mo_name(rng), mo_phone(rng), hk_mo_email(rng),
                  mo_address(rng), dialect="mo")


REGION_TEMPLATES = {
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


# =============================================================================
# Main
# =============================================================================

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSONL path (one record per line).")
    parser.add_argument("--n", type=int, default=1000,
                        help="Number of examples to generate (split evenly across regions).")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--regions", nargs="+", default=["tw", "hk", "mo"],
                        choices=["tw", "hk", "mo"],
                        help="Which regions to include.")
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    records: list[dict] = []
    regions = args.regions
    per_region = args.n // len(regions)
    remainder = args.n - per_region * len(regions)

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
                    "id":     f"{region}_synth_{i:04d}",
                    "source": "scripts.gen_zhyt_test_data",
                    "region": REGION_LABEL[region],
                },
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Per-region counts.
    from collections import Counter
    region_counts = Counter(r["info"]["region"] for r in records)
    label_counts = Counter()
    for r in records:
        for k in r["spans"]:
            label_counts[k.split(": ", 1)[0]] += 1

    print(f"wrote {len(records)} records to {args.out}")
    print(f"  per region: {dict(region_counts)}")
    print(f"  per label : {dict(label_counts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())