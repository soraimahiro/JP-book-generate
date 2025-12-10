from PIL import Image, ImageDraw, ImageFont, ImageFilter
import json
import random
import numpy as np

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

def create_aged_paper_background(width, height):
    """創建模擬古書掃描的背景"""
    # 創建基礎紙張顏色 (米黃色)
    base_color = random.randint(235, 250)
    img_array = np.full((height, width, 3), base_color, dtype=np.uint8)
    
    # 1. 添加大範圍的色調變化（模擬紙張不均勻）
    for _ in range(5):
        center_x = random.randint(0, width)
        center_y = random.randint(0, height)
        radius = random.randint(200, 500)
        intensity = random.randint(-20, 10)
        
        y_indices, x_indices = np.ogrid[:height, :width]
        distances = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
        mask = np.clip(1 - distances / radius, 0, 1)
        
        for c in range(3):
            adjustment = (mask * intensity).astype(np.int16)
            img_array[:, :, c] = np.clip(img_array[:, :, c].astype(np.int16) + adjustment, 0, 255).astype(np.uint8)
    
    # 2. 添加中等大小的污漬
    for _ in range(random.randint(10, 20)):
        center_x = random.randint(0, width)
        center_y = random.randint(0, height)
        radius = random.randint(30, 150)
        intensity = random.randint(-40, -10)
        
        y_indices, x_indices = np.ogrid[:height, :width]
        distances = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
        mask = np.clip(1 - distances / radius, 0, 1) ** 2
        
        for c in range(3):
            adjustment = (mask * intensity).astype(np.int16)
            img_array[:, :, c] = np.clip(img_array[:, :, c].astype(np.int16) + adjustment, 0, 255).astype(np.uint8)
    
    # 3. 添加細小噪點（模擬紙張纖維和灰塵）
    noise = np.random.randint(-15, 15, (height, width, 3), dtype=np.int16)
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 4. 添加一些小黑點（模擬墨漬）
    num_spots = random.randint(50, 150)
    for _ in range(num_spots):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        spot_size = random.randint(1, 3)
        darkness = random.randint(0, 100)
        
        for dy in range(-spot_size, spot_size + 1):
            for dx in range(-spot_size, spot_size + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if dx*dx + dy*dy <= spot_size*spot_size:
                        img_array[ny, nx] = np.maximum(img_array[ny, nx] - darkness, 0)
    
    # 轉換為 PIL Image
    img = Image.fromarray(img_array, mode='RGB')
    
    # 5. 應用輕微的高斯模糊，使噪點更自然
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    
    return img

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
    
    # 建立圖片（垂直排列）- 使用古書背景
    img = create_aged_paper_background(900, 1200)
    write = ImageDraw.Draw(img)
    
    # 儲存所有標註
    all_annotations = []
    
    # 繪製垂直排列的漢字
    x_borders = list(range(50, 800, 90))
    y_borders = list(range(80, 1100, 55))
    
    for x_border in x_borders:
        x_pos = x_border + 25
        for i, y_pos in enumerate(y_borders):
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
        
        write.rectangle((x_border, 50, x_border + 90, 1150), outline=(0, 0, 0), width=2)
    
    # 儲存圖片
    img.save("new.jpg")
    
    with open('ground_truth.json', 'w', encoding='utf-8') as f:
        json.dump(all_annotations, f, ensure_ascii=False, indent=2)
    
    print(f"\nsaved new.jpg")
    print(f"Ground Truth saved as ground_truth.json")
    print(f"Total annotations: {len(all_annotations)}")