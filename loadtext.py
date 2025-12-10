import pandas as pd
import json

# Read the specific sheet from the Excel file
df = pd.read_excel('教育部4808個常用字.xls', sheet_name='教育部4808個常用字')

# Extract the "常用字" column and convert to list
common_chars = df['常用字'].tolist()

# Combine all lists into one dictionary
data = {
    'common_chars': common_chars,
    'kana': ['ア', 'イ', 'ウ', 'エ', 'オ', 'カ', 'キ', 'ク', 'ケ', 'コ', 'サ', 'シ', 'ス', 'セ', 'ソ', 'タ', 'チ', 'ツ', 'テ', 'ト', 'ナ', 'ニ', 'ヌ', 'ネ', 'ノ', 'ハ', 'ヒ', 'フ', 'ヘ', 'ホ', 'マ', 'ミ', 'ム', 'メ', 'モ', 'ヤ', 'ユ', 'ヨ', 'ラ', 'リ', 'ル', 'レ', 'ロ', 'ワ', 'ヲ', 'ン', '上', '下', '一', '二'],
    'symbols': ["丨", "〇"]
}

# Save to JSON file
with open('common_char.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False)

print(f"Successfully saved {len(common_chars)} common characters, {len(data['kana'])} kana, and {len(data['symbols'])} symbols to common_char.json")
