import grequests
import json

url = 'http://ooxx:614/inference'

adata = []
adata.append(json.dumps({
    "esun_uuid": "6465e4f10a0b099c44791e901efb7e8d9268b972b95b6fa53db93780b6b22fbf",
    "esun_timestamp": 1590493849,
    "sentence_list":
    [
        "在 可 成家 範圍 內 將 部分 成交",
        "在 可 成家 範圍 內 將 部份 成交",
        "再 可 成家 範圍 內 將 部分 成交",
        "在 可成 家 範圍 內 將 部分 成交",
        "在 可 陳 家 範圍 內 將 部分 成交",
        "在 可成 加 範圍 內 將 部分 成交",
        "再 可 成家 範圍 內 將 部份 成交",
        "在 可 陳 嘉 範圍 內 將 部分 成交",
        "在 可 成交 範圍 內 將 部分 成交",
        "在 可成 家 範圍 內 將 部份 成交"
    ],
    "phoneme_sequence_list":
    [
        "ts aI4 k_h ax3 ttss_h ax N2 ts6 j A:1 f A: n4 w eI2 n eI4 ts6 j A: N1 p u:4 f ax n4 ttss_h ax N2 ts6 j aU1",
        "ts aI4 k_h ax3 ttss_h ax N2 ts6 j A:1 f A: n4 w eI2 n eI4 ts6 j A: N1 p u:4 f ax n4 ttss_h ax N2 ts6 j aU1",
        "ts aI4 k_h ax3 ttss_h ax N2 ts6 j A:1 f A: n4 w eI2 n eI4 ts6 j A: N1 p u:4 f ax n4 ttss_h ax N2 ts6 j aU1",
        "ts aI4 k_h ax3 ttss_h ax n2 ts6 j A:1 f A: n4 w eI2 n eI4 ts6 j A: N1 p u:4 f ax n4 ttss_h ax N2 ts6 j aU1",
        "ts aI4 k_h ax3 ttss_h ax N2 ts6 j aU1 f A: n4 w eI2 n eI4 ts6 j A: N1 p u:4 f ax n4 ttss_h ax N2 ts6 j aU1",
        "ts aI4 k_h ax3 ttss_h ax N2 ts6 j A:1 f A: n4 w eI2 n eI4 ts6 j A: N1 p u:4 f ax n4 ttss_h ax N2 ts6 j aU1",
        "ts aI4 k_h ax3 ttss_h ax N2 ts6 j A:1 f A: n4 w eI2 n eI4 ts6 j A: N1 p u:4 f ax n4 ttss_h ax N2 ts6 j aU1",
        "ts aI4 k_h ax3 ttss_h ax n2 ts6 j A:1 f A: n4 w eI2 n eI4 ts6 j A: N1 p u:4 f ax n4 ttss_h ax N2 ts6 j aU1",
        "ts aI4 k_h ax3 ts_h ax N2 ts6 j A:1 f A: n4 w eI2 n eI4 ts6 j A: N1 p u:4 f ax n4 ttss_h ax N2 ts6 j aU1",
        "ts aI4 k_h ax3 ts_h A: n1 ts6 j A:1 f A: n4 w eI2 n eI4 ts6 j A: N1 p u:4 f ax n4 ttss_h ax N2 ts6 j aU1"
    ],
    "retry": 2
}))

adata.append(json.dumps({
    "esun_uuid": "6465e4f10a0b099c44791e901efb7e8d9268b972b95b6fa53db93780b6b22fbf",
    "esun_timestamp": 1590493849,
    "sentence_list": ["可能 導致 不是 泡沫 再現", "可能 導致 不是 泡沫 在線", "可能 導致 不是 泡沫 再 見", "可能 導致 不是 泡沫 在 現", "可能 導致 不 是 泡沫 再現", "可能 導致 不是 泡沫 在 見", "可能 導致 不是 泡沫 在 限", "可能 導致 不 是 泡沫 在線", "可能 導致 不是 泡沫 在 線", "可能 導致 股市 泡沫 再現"],
    "phoneme_sequence_list": ["k_h ax3 n ax N2 t aU3 ttss4 p u:2 ss4 p_h aU4 m O:4 ts aI4 s6 j A: n4", "k_h ax3 n ax N2 t aU3 ttss4 p u:2 ss4 p_h aU4 m O:4 ts aI4 s6 j A: n4", "k_h ax3 n ax N2 t aU3 ttss4 p u:2 ss4 p_h aU4 m O:4 ts aI4 s6 j A: n4", "k_h ax3 n ax N2 t aU3 ttss4 p u:2 ss4 p_h aU4 m O:4 ts aI4 s6 j A: n4", "k_h ax3 n ax N2 t aU3 ttss4 p u:2 ss4 p_h aU4 m O:4 ts aI4 s6 j A: n4", "k_h ax3 n ax N2 t aU3 ttss4 p u:2 ss4 p_h aU4 m O:4 ts aI4 s6 j A: n4", "k_h ax3 n ax N2 t aU3 ttss4 p u:2 ss4 p_h aU4 m O:4 ts aI4 s6 j A: n4", "k_h ax3 n ax N2 t aU3 ttss4 k u:3 ss4 p_h aU4 m O:4 ts aI4 s6 j A: n4", "k_h ax3 n ax N2 t aU3 ttss4 p u:2 ss4 p_h aU4 m O:4 ts aI4 s6 j A: n4", "k_h ax3 n ax N2 t aU3 ttss4 p u:2 ss4 p_h aU4 m O:4 ts aI4 s6 j A: n4"],
    "retry": 2
}))

adata.append(json.dumps({
    "esun_uuid": "6465e4f10a0b099c44791e901efb7e8d9268b972b95b6fa53db93780b6b22fbf",
    "esun_timestamp": 1590493849,
    "sentence_list": ["我 發現 新 元 噢 許多 天使 投資 機構 不同 的 地方", "我 發現 新原 噢 許多 天使 投資 機構 不同 的 地方", "我 發現 新 元 於 許多 天使 投資 機構 不同 的 地方", "我 發現 新原 於 許多 天使 投資 機構 不同 的 地方", "我 發現 新原 與 許多 天使 投資 機構 不同 的 地方", "我 發現 新 元 與 許多 天使 投資 機構 不同 的 地方", "我 發現 昕 媛 噢 許多 天使 投資 機構 不同 的 地方", "我 發現 昕 媛 於 許多 天使 投資 機構 不同 的 地方", "我 發現 新 元 噢 許多 天時 投資 機構 不同 的 地方", "我 發現 新原 噢 許多 天時 投資 機構 不同 的 地方"],
    "phoneme_sequence_list": ["w O:3 f A:1 s6 j A: n4 s6 j ax n1 H A: n2 y3 s6 y3 t w O:1 t_h j A: n1 ss3 t_h oU2 ts1 ts6 i:1 k oU4 p u:4 t_h w ax N2 t ax5 t i:4 f A: N1", "w O:3 f A:1 s6 j A: n4 s6 j ax n1 H A: n2 y3 s6 y3 t w O:1 t_h j A: n1 ss3 t_h oU2 ts1 ts6 i:1 k oU4 p u:4 t_h w ax N2 t ax5 t i:4 f A: N1", "w O:3 f A:1 s6 j A: n4 s6 j ax n1 H A: n2 y2 s6 y3 t w O:1 t_h j A: n1 ss3 t_h oU2 ts1 ts6 i:1 k oU4 p u:4 t_h w ax N2 t ax5 t i:4 f A: N1", "w O:3 f A:1 s6 j A: n4 s6 j ax n1 H A: n2 y2 s6 y3 t w O:1 t_h j A: n1 ss3 t_h oU2 ts1 ts6 i:1 k oU4 p u:4 t_h w ax N2 t ax5 t i:4 f A: N1", "w O:3 f A:1 s6 j A: n4 s6 j ax n1 H A: n2 y3 s6 y3 t w O:1 t_h j A: n1 ss3 t_h oU2 ts1 ts6 i:1 k oU4 p u:4 t_h w ax N2 t ax5 t i:4 f A: N1", "w O:3 f A:1 s6 j A: n4 s6 j ax n1 H A: n2 y2 s6 y3 t w O:1 t_h j A: n1 ss3 t_h oU2 ts1 ts6 i:1 k oU4 p u:4 t_h w ax N2 t ax5 t i:4 f A: N1", "w O:3 f A:1 s6 j A: n4 s6 j ax n1 H A: n2 y3 s6 y3 t w O:1 t_h j A: n1 ss2 t_h oU2 ts1 ts6 i:1 k oU4 p u:4 t_h w ax N2 t ax5 t i:4 f A: N1", "w O:3 f A:1 s6 j A: n4 s6 j ax n1 H A: n2 y3 s6 y3 t w O:1 t_h j A: n1 ss2 t_h oU2 ts1 ts6 i:1 k oU4 p u:4 t_h w ax N2 t ax5 t i:4 f A: N1", "w O:3 f A:1 s6 j A: n4 s6 j ax n1 H A: n2 y2 s6 y3 t w O:1 t_h j A: n1 ss2 t_h oU2 ts1 ts6 i:1 k oU4 p u:4 t_h w ax N2 t ax5 t i:4 f A: N1", "w O:3 f A:1 s6 j A: n4 s6 j ax n1 H A: n2 y2 s6 y3 t w O:1 t_h j A: n1 ss2 t_h oU2 ts1 ts6 i:1 k oU4 p u:4 t_h w ax N2 t ax5 t i:4 f A: N1"],
    "retry": 2
}))

adata.append(json.dumps({
    "esun_uuid": "6465e4f10a0b099c44791e901efb7e8d9268b972b95b6fa53db93780b6b22fbf",
    "esun_timestamp": 1590493849,
    "sentence_list": ["我 發現 新 元 噢 許多 天使 投資 機構 不同 的 地方", "我 發現 新原 噢 許多 天使 投資 機構 不同 的 地方", "我 發現 新 元 於 許多 天使 投資 機構 不同 的 地方", "我 發現 新原 於 許多 天使 投資 機構 不同 的 地方", "我 發現 新原 與 許多 天使 投資 機構 不同 的 地方", "我 發現 新 元 與 許多 天使 投資 機構 不同 的 地方", "我 發現 昕 媛 噢 許多 天使 投資 機構 不同 的 地方", "我 發現 昕 媛 於 許多 天使 投資 機構 不同 的 地方", "我 發現 新 元 噢 許多 天時 投資 機構 不同 的 地方", "我 發現 新原 噢 許多 天時 投資 機構 不同 的 地方"],
    "phoneme_sequence_list": ["w O:3 f A:1 s6 j A: n4 s6 j ax n1 H A: n2 y3 s6 y3 t w O:1 t_h j A: n1 ss3 t_h oU2 ts1 ts6 i:1 k oU4 p u:4 t_h w ax N2 t ax5 t i:4 f A: N1", "w O:3 f A:1 s6 j A: n4 s6 j ax n1 H A: n2 y3 s6 y3 t w O:1 t_h j A: n1 ss3 t_h oU2 ts1 ts6 i:1 k oU4 p u:4 t_h w ax N2 t ax5 t i:4 f A: N1", "w O:3 f A:1 s6 j A: n4 s6 j ax n1 H A: n2 y2 s6 y3 t w O:1 t_h j A: n1 ss3 t_h oU2 ts1 ts6 i:1 k oU4 p u:4 t_h w ax N2 t ax5 t i:4 f A: N1", "w O:3 f A:1 s6 j A: n4 s6 j ax n1 H A: n2 y2 s6 y3 t w O:1 t_h j A: n1 ss3 t_h oU2 ts1 ts6 i:1 k oU4 p u:4 t_h w ax N2 t ax5 t i:4 f A: N1", "w O:3 f A:1 s6 j A: n4 s6 j ax n1 H A: n2 y3 s6 y3 t w O:1 t_h j A: n1 ss3 t_h oU2 ts1 ts6 i:1 k oU4 p u:4 t_h w ax N2 t ax5 t i:4 f A: N1", "w O:3 f A:1 s6 j A: n4 s6 j ax n1 H A: n2 y2 s6 y3 t w O:1 t_h j A: n1 ss3 t_h oU2 ts1 ts6 i:1 k oU4 p u:4 t_h w ax N2 t ax5 t i:4 f A: N1", "w O:3 f A:1 s6 j A: n4 s6 j ax n1 H A: n2 y3 s6 y3 t w O:1 t_h j A: n1 ss2 t_h oU2 ts1 ts6 i:1 k oU4 p u:4 t_h w ax N2 t ax5 t i:4 f A: N1", "w O:3 f A:1 s6 j A: n4 s6 j ax n1 H A: n2 y3 s6 y3 t w O:1 t_h j A: n1 ss2 t_h oU2 ts1 ts6 i:1 k oU4 p u:4 t_h w ax N2 t ax5 t i:4 f A: N1", "w O:3 f A:1 s6 j A: n4 s6 j ax n1 H A: n2 y2 s6 y3 t w O:1 t_h j A: n1 ss2 t_h oU2 ts1 ts6 i:1 k oU4 p u:4 t_h w ax N2 t ax5 t i:4 f A: N1", "w O:3 f A:1 s6 j A: n4 s6 j ax n1 H A: n2 y2 s6 y3 t w O:1 t_h j A: n1 ss2 t_h oU2 ts1 ts6 i:1 k oU4 p u:4 t_h w ax N2 t ax5 t i:4 f A: N1"],
    "retry": 2
}))
adata.append(json.dumps({
    "esun_uuid": "6465e4f10a0b099c44791e901efb7e8d9268b972b95b6fa53db93780b6b22fbf",
    "esun_timestamp": 1590493849,
    "sentence_list": ["並 提升 內部 監督 機制", "並 提升 那 不 見 都 機制", "並 提升 內部 間 都 機制", "並 提升 內部 件 都 機制", "並 提升 那 不 間 都 機制", "並 提升 那 不 監督 機制", "並 提升 那 部 監督 機制", "並 提升 那 布建 都 機制", "並 提升 內 不 見 都 機制", "並 提升 那 不 件 都 機制"],
    "phoneme_sequence_list": ["p j ax N4 t_h i:2 ss ax N1 n eI4 p u:4 ts6 j A: n1 t u:1 ts6 i:1 ttss4", "p j ax N4 t_h i:2 ss ax N1 n eI4 p u:4 ts6 j A: n4 t u:1 ts6 i:1 ttss4", "p j ax N4 t_h i:2 ss ax N1 n eI4 p u:4 ts6 j A: n4 t u:1 ts6 i:1 ttss4", "p j ax N4 t_h i:2 ss ax N1 n eI4 p u:4 ts6 j A: n4 t u:1 ts6 i:1 ttss4", "p j ax N4 t_h i:2 ss ax N1 n eI4 p u:4 ts6 j A: n4 t u:1 ts6 i:1 ttss4", "p j ax N4 t_h i:2 ss ax N1 n eI4 p u:4 ts6 j A: n1 t u:1 ts6 i:1 ttss4", "p j ax N4 t_h i:2 ss ax N1 n eI4 p u:4 ts6 j A: n1 t u:1 ts6 i:1 ttss4", "p j ax N4 t_h i:2 ss ax N1 n eI4 p u:4 ts6 j A: n4 t u:1 ts6 i:1 ttss4", "p j ax N4 t_h i:2 ss ax N1 n eI4 p u:4 ts6 j A: n4 t u:1 ts6 i:1 ttss4", "p j ax N4 t_h i:2 ss ax N1 n eI4 p u:4 ts6 j A: n4 t u:1 ts6 i:1 ttss4"],
    "retry": 2
}))
adata.append(json.dumps({
    "esun_uuid": "6465e4f10a0b099c44791e901efb7e8d9268b972b95b6fa53db93780b6b22fbf",
    "esun_timestamp": 1590493849,
    "sentence_list": ["央行 進場 買 回調 接", "央行 進場 買回 掉 接", "央行 進場 買回 調 接", "央行 進場 買 回調 結", "央行 進場 買 回調 皆", "央行 進場 買 回調 街", "央行 進場 買回 掉 結", "央行 進場 買回 掉 皆", "央行 進場 買回 調 結", "央行 進場 買回 掉 街"],
    "phoneme_sequence_list": ["j A: N1 x A: N2 ts6 j ax n4 ttss_h A: N3 m aI3 x w eI2 t j aU4 ts6 j E1", "j A: N1 x A: N2 ts6 j ax n4 ttss_h A: N3 m aI3 x w eI2 t j aU4 ts6 j E1", "j A: N1 x A: N2 ts6 j ax n4 ttss_h A: N3 m aI3 x w eI2 t j aU4 ts6 j E1", "j A: N1 x A: N2 ts6 j ax n4 ttss_h A: N3 m aI3 x w eI2 t j aU4 ts6 j E1", "j A: N1 x A: N2 ts6 j ax n4 ttss_h A: N3 m aI3 x w eI2 t j aU4 ts6 j E1", "j A: N1 x A: N2 ts6 j ax n4 ttss_h A: N3 m aI3 x w eI2 t j aU4 ts6 j E1", "j A: N1 x A: N2 ts6 j ax n4 ttss_h A: N3 m aI3 x w eI2 t j aU4 ts6 j E1", "j A: N1 x A: N2 ts6 j ax n4 ttss_h A: N3 m aI3 x w eI2 t j aU4 ts6 j E1", "j A: N1 x A: N2 ts6 j ax n4 ttss_h A: N3 m aI3 x w eI2 t j aU4 ts6 j E1", "j A: N1 x A: N2 ts6 j ax n4 ttss_h A: N3 m aI3 x w eI2 t j aU4 ts6 j E1"],
    "retry": 2
}))
adata.append(json.dumps({
    "esun_uuid": "6465e4f10a0b099c44791e901efb7e8d9268b972b95b6fa53db93780b6b22fbf",
    "esun_timestamp": 1590493849,
    "sentence_list": ["吸引 不少 轉單", "吸引 不少 轉 單", "吸引 不少 簡單"],
    "phoneme_sequence_list": ["s6 i:1 j ax n3 p u:4 ss aU3 ttss w A: n3 t A: n1", "s6 i:1 j ax n3 p u:4 ss aU3 ttss w A: n3 t A: n1", "s6 i:1 j ax n3 p u:4 ss aU3 ts6 j A: n3 t A: n1"],
    "retry": 2
}))
adata.append(json.dumps({
    "esun_uuid": "6465e4f10a0b099c44791e901efb7e8d9268b972b95b6fa53db93780b6b22fbf",
    "esun_timestamp": 1590493849,
    "sentence_list": ["車 是 已 逐漸 回溫", "車室 已 逐漸 回溫", "車市 已 逐漸 回溫", "車 是 一 逐漸 回溫", "車 是以 逐漸 回溫", "車式 已 逐漸 回溫", "車 是否 已 逐漸 回溫", "車 是 以 逐漸 回溫", "車室 一 逐漸 回溫", "車市 一 逐漸 回溫"],
    "phoneme_sequence_list": ["ttss_h ax1 ss4 i:3 ttss u:2 ts6 j A: n4 x w eI2 w ax n1", "ttss_h ax1 ss4 i:3 ttss u:2 ts6 j A: n4 x w eI2 w ax n1", "ttss_h ax1 ss4 i:4 ttss u:2 ts6 j A: n4 x w eI2 w ax n1", "ttss_h ax1 ss4 i:3 ttss u:2 ts6 j A: n4 x w eI2 w ax n1", "ttss_h ax1 ss4 f oU3 i:3 ttss u:2 ts6 j A: n4 x w eI2 w ax n1", "ttss_h ax1 ss4 i:3 ttss u:2 ts6 j A: n4 x w eI2 w ax n1", "ttss_h ax1 ss4 i:4 ttss u:2 ts6 j A: n4 x w eI2 w ax n1", "ttss_h ax1 ss4 i:3 ttss u:2 ts6 j A: n4 x w eI2 w ax n1", "ts_h ax4 ss4 i:3 ttss u:2 ts6 j A: n4 x w eI2 w ax n1", "ttss ax4 ss4 i:3 ttss u:2 ts6 j A: n4 x w eI2 w ax n1"],
    "retry": 2
}))

adata.append(json.dumps({
    "esun_uuid": "6465e4f10a0b099c44791e901efb7e8d9268b972b95b6fa53db93780b6b22fbf",
    "esun_timestamp": 1590493849,
    "sentence_list": ["不只 高雄 市長 韓國 瑜 為了 衝刺 選舉", "不止 高雄 市長 韓國 瑜 為了 衝刺 選舉", "不只 高 雄 市長 韓國 瑜 為了 衝刺 選舉", "不知 高雄 市長 韓國 瑜 為了 衝刺 選舉", "不止 高 雄 市長 韓國 瑜 為了 衝刺 選舉", "不只 高雄 市長 韓國 於 為了 衝刺 選舉", "簿子 高雄 市長 韓國 瑜 為了 衝刺 選舉", "不值 高雄 市長 韓國 瑜 為了 衝刺 選舉", "不知 高 雄 市長 韓國 瑜 為了 衝刺 選舉", "不只 高雄 市長 韓 國 愉 為了 衝刺 選舉"],
    "phoneme_sequence_list": ["p u:4 ttss3 k aU1 s6 H ax N2 ss4 ttss A: N3 x A: n2 k w O:2 y2 w eI4 l ax5 ttss_h w ax N1 ts_h4 s6 H A: n3 ts6 y3", "p u:4 ttss3 k aU1 s6 H ax N2 ss4 ttss A: N3 x A: n2 k w O:2 y2 w eI4 l ax5 ttss_h w ax N1 ts_h4 s6 H A: n3 ts6 y3", "p u:4 ttss3 k aU1 s6 H ax N2 ss4 ttss A: N3 x A: n2 k w O:2 y2 w eI4 l ax5 ttss_h w ax N1 ts_h4 s6 H A: n3 ts6 y3", "p u:4 ttss1 k aU1 s6 H ax N2 ss4 ttss A: N3 x A: n2 k w O:2 y2 w eI4 l ax5 ttss_h w ax N1 ts_h4 s6 H A: n3 ts6 y3", "p u:4 ttss3 k aU1 s6 H ax N2 ss4 ttss A: N3 x A: n2 k w O:2 y2 w eI4 l ax5 ttss_h w ax N1 ts_h4 s6 H A: n3 ts6 y3", "p u:4 ts5 k aU1 s6 H ax N2 ss4 ttss A: N3 x A: n2 k w O:2 y2 w eI4 l ax5 ttss_h w ax N1 ts_h4 s6 H A: n3 ts6 y3", "p u:4 ttss2 k aU1 s6 H ax N2 ss4 ttss A: N3 x A: n2 k w O:2 y2 w eI4 l ax5 ttss_h w ax N1 ts_h4 s6 H A: n3 ts6 y3", "p u:4 ttss1 k aU1 s6 H ax N2 ss4 ttss A: N3 x A: n2 k w O:2 y2 w eI4 l ax5 ttss_h w ax N1 ts_h4 s6 H A: n3 ts6 y3", "p u:4 ttss3 k aU1 s6 H ax N2 ss4 ttss A: N3 x A: n2 k w O:2 y2 w eI4 l ax5 ttss_h w ax N1 ts_h4 s6 H A: n3 ts6 y3", "p u:4 ttss3 k aU1 s6 H ax N2 ss4 ttss A: N3 x A: n2 k w O:2 y2 w eI4 l ax5 ttss_h w ax N1 ts_h4 s6 H A: n3 ts6 y3"],
    "retry": 2
}))

adata.append(json.dumps({
    "esun_uuid": "6465e4f10a0b099c44791e901efb7e8d9268b972b95b6fa53db93780b6b22fbf",
    "esun_timestamp": 1590493849,
    "sentence_list": ["分期 登錄", "分期 登入", "分歧 登錄", "分期 登陸", "分期 要 登錄"],
    "phoneme_sequence_list": ["f ax n1 ts6_h i:2 t ax N1 l u:4", "f ax n1 ts6_h i:2 t ax N1 zz u:4", "f ax n1 ts6_h i:2 t ax N1 l u:4", "f ax n1 ts6_h i:2 t ax N1 l u:4", "f ax n1 ts6_h i:2 j aU1 t ax N1 l u:4"],
    "retry": 2
}))

# print(adata)

header = {"Content-type": "application/json", "Accept": "application/json"}

N = 10
rs = []
for i in range(N):
    req = grequests.request("POST", url=url, data=adata[i], headers=header)
    rs.append(req)

#req = grequests.request("POST", url=url, data=adata[0], headers=header)
#rs = [req for _ in range(N)]

#req = grequests.request("POST", url=url2, data=adata, headers=header)
# for _ in range(N):
#   rs.append(req)

#sizeT = int(input())
sizeT = 10
resp = grequests.map(rs, size=sizeT, gtimeout=10)

for i, item in enumerate(resp):
    try:
        print(item)
        #content = json.loads((item.content).decode('utf8'))

        # print(content['answer'])
        # print(content['server_timestamp'])
    except:
        continue

    finally:

        print("--<{}>--".format(i + 1))
