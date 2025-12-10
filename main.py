from PIL import Image, ImageDraw, ImageFont
import json
import random

all_chars = None

def load_character(type):
    global all_chars
    if all_chars is None:
        with open('common_char.json', 'r', encoding='utf-8') as f:
            all_chars = json.load(f)

    if type not in all_chars:
        raise ValueError(f"Type '{type}' not found in character data.")
    return random.choice(all_chars[type])

SYMBOLS = ['|', '○']

def add_annotations(char_x, char_y, char_size, write, have_symbol, annotations):
    kana_font = [ImageFont.truetype('font/NotoSerifTC-Bold.ttf', char_size // 3, encoding='utf-8'),
                 ImageFont.truetype('font/NotoSerifTC-Bold.ttf', char_size // 3 + 3, encoding='utf-8'),
                 ImageFont.truetype('font/NotoSerifTC-Regular.ttf', char_size // 3 + 3, encoding='utf-8'),
                 ImageFont.truetype('font/NotoSerifTC-Regular.ttf', char_size // 3 + 5, encoding='utf-8')]
    
    positions = {
        'top_left'    : (char_x - 12           , char_y + 5),
        'top_right'   : (char_x + char_size - 3, char_y + 5),
        'bottom_left' : (char_x - 12           , char_y + char_size + 3),
        'bottom_right': (char_x + char_size - 3, char_y + char_size + 3),
        'right'       : (char_x + char_size - 3, char_y + char_size // 2)
    }
    
    for direction, (base_x, base_y) in positions.items():
        # 50% 機率插入假名
        if random.random() < 0.3:
            kana_text = load_character('kana')
            
            # 繪製假名
            font = random.choice(kana_font)
            write.text((base_x, base_y), kana_text, (0, 0, 0), font=font)
            
            # 計算 bounding box
            bbox = font.getbbox(kana_text)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            annotations.append({
                'type': 'kana',
                'text': kana_text,
                'direction': direction,
                'bbox': [base_x, base_y, base_x + width, base_y + height]
            })
    
    # 漢字下方符號 (50% 機率)
    if have_symbol and random.random() < 0.5:
        symbol = load_character('symbols')
        symbol_x = char_x + char_size // 2 - 10
        symbol_y = char_y + char_size + 6
        
        font = kana_font[2]
        write.text((symbol_x, symbol_y), symbol, (0, 0, 0), font=font)
        
        # 計算 bounding box
        bbox = font.getbbox(symbol)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        annotations.append({
            'type': 'symbol',
            'text': symbol,
            'bbox': [symbol_x, symbol_y, symbol_x + width, symbol_y + height]
        })

if __name__ == "__main__":
    # 字型設定
    kanji_font = ImageFont.truetype('font/NotoSerifTC-Regular.ttf', 40, encoding='utf-8')
    
    # 建立圖片（垂直排列）
    img = Image.new("RGB", (900, 1200), (255, 255, 255))
    write = ImageDraw.Draw(img)
    write.rectangle((50, 50, 850, 1150), outline=(0, 0, 0), width=3)
    
    # 儲存所有標註
    all_annotations = []
    
    # 繪製垂直排列的漢字
    x_positions = list(range(80, 800, 90))
    y_positions = list(range(80, 1100, 55))
    
    for x_pos in x_positions:
        for i, y_pos in enumerate(y_positions):
            write_char = load_character('common_chars')
            
            # 繪製漢字
            write.text((x_pos, y_pos), write_char, (0, 0, 0), font=kanji_font)
            
            # 添加假名和符號
            char_annotations = []
            add_annotations(x_pos, y_pos, 40, write, True, char_annotations)
            
            print(f"{write_char} at ({x_pos}, {y_pos})")
            for ann in char_annotations:
                print(f"  - {ann['type']:<6s}: {ann['text']} at {ann['bbox']}")
            all_annotations = all_annotations + char_annotations
        
        write.line((x_pos + 65, 50, x_pos + 65, 1150), fill=(0, 0, 0), width=2)
    
    # 儲存圖片
    img.save("new.jpg")
    
    with open('ground_truth.json', 'w', encoding='utf-8') as f:
        json.dump(all_annotations, f, ensure_ascii=False, indent=2)
    
    print(f"\nsaved new.jpg")
    print(f"Ground Truth saved as ground_truth.json")
    print(f"Total annotations: {len(all_annotations)}")