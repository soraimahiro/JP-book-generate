from PIL import Image, ImageDraw, ImageFont, ImageFilter
import json
import random
import numpy as np
from tqdm import tqdm
import cv2

all_chars = None
# 快取字型物件，避免重複載入
kanji_font_cache = None
kana_font_cache = None

def load_character(type):
    global all_chars
    if all_chars is None:
        with open('common_char.json', 'r', encoding='utf-8') as f:
            all_chars = json.load(f)

    if type not in all_chars:
        raise ValueError(f"Type '{type}' not found in character data.")
    return random.choice(all_chars[type])

def get_fonts():
    """快取字型物件"""
    global kanji_font_cache, kana_font_cache
    if kanji_font_cache is None:
        kanji_font_cache = ImageFont.truetype('font/NotoSerifTC-Regular.ttf', 40, encoding='utf-8')
    if kana_font_cache is None:
        kana_font_cache = [
            ImageFont.truetype('font/YujiSyuku-Regular.ttf', 16, encoding='utf-8'),
            ImageFont.truetype('font/YujiSyuku-Regular.ttf', 16, encoding='utf-8'),
            ImageFont.truetype('font/KleeOne-SemiBold.ttf', 18, encoding='utf-8'),
            ImageFont.truetype('font/KleeOne-SemiBold.ttf', 18, encoding='utf-8')
        ]
    return kanji_font_cache, kana_font_cache

def create_aged_paper_background(width, height):
    """創建模擬古書掃描的背景"""
    base_color = random.randint(235, 255)
    img_array = np.full((height, width, 3), base_color, dtype=np.uint8)
    
    # 預先計算網格座標
    y_indices, x_indices = np.ogrid[:height, :width]
    
    # 1. 添加大範圍的色調變化（向量化操作）
    for _ in range(5):
        center_x = random.randint(0, width)
        center_y = random.randint(0, height)
        radius = random.randint(200, 500)
        intensity = random.randint(-20, 10)
        
        distances = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
        mask = np.clip(1 - distances / radius, 0, 1)
        
        # 一次性處理所有通道
        adjustment = (mask * intensity).astype(np.int16)
        img_array = np.clip(img_array.astype(np.int16) + adjustment[:, :, np.newaxis], 0, 255).astype(np.uint8)
    
    # 2. 添加中等大小的污漬
    for _ in range(random.randint(10, 20)):
        center_x = random.randint(0, width)
        center_y = random.randint(0, height)
        radius = random.randint(30, 150)
        intensity = random.randint(-40, -10)
        
        distances = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
        mask = np.clip(1 - distances / radius, 0, 1) ** 2
        
        adjustment = (mask * intensity).astype(np.int16)
        img_array = np.clip(img_array.astype(np.int16) + adjustment[:, :, np.newaxis], 0, 255).astype(np.uint8)
    
    # 3. 添加細小噪點
    noise = np.random.randint(-15, 15, (height, width, 3), dtype=np.int16)
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 4. 添加小黑點 - 使用向量化操作
    num_spots = random.randint(50, 200)
    spot_coords = np.random.randint(0, [width, height], size=(num_spots, 2))
    spot_sizes = np.random.randint(1, 4, size=num_spots)
    darkness_values = np.random.randint(10, 151, size=num_spots)
    
    for (x, y), spot_size, darkness in zip(spot_coords, spot_sizes, darkness_values):
        y1, y2 = max(0, y - spot_size), min(height, y + spot_size + 1)
        x1, x2 = max(0, x - spot_size), min(width, x + spot_size + 1)
        img_array[y1:y2, x1:x2] = np.maximum(img_array[y1:y2, x1:x2] - darkness, 0)
    
    # 使用 cv2 進行高斯模糊（比 PIL 快）
    img_array = cv2.GaussianBlur(img_array, (3, 3), 0.8)
    
    img = Image.fromarray(img_array, mode='RGB')
    return img

def add_annotations(center_x, center_y, char_size, write, have_symbol, annotations, prob_kana=0.3):
    """在兩個字中間插入假名和符號"""
    _, kana_fonts = get_fonts()
    
    directions = {
        'left': center_x - char_size // 2,
        'right': center_x + char_size // 2
    }
    
    for direction, base_x in directions.items():
        if random.random() < prob_kana:
            num_kana = 2 if random.random() < 0.2 else 1
            
            if num_kana == 1:
                font = random.choice(kana_fonts)
                kana_text = load_character('kana')
                
                bbox = font.getbbox(kana_text)
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]

                x_pos = base_x
                if direction == 'left':
                    x_pos -= width
                    x_pos += random.randint(-3, 1)
                else:
                    x_pos += random.randint(-1, 3)
                y_pos = center_y + random.randint(-3, 3)
                
                write.text((x_pos, y_pos), kana_text, (0, 0, 0), font=font)
                
                annotations.append({
                    'type': 'kana',
                    'text': kana_text,
                    'bbox': [x_pos, y_pos, x_pos + width, y_pos + height]
                })
            else:
                # 兩個假名
                for is_down in [False, True]:
                    kana_text = load_character('kana')
                    font = random.choice(kana_fonts)
                    bbox = font.getbbox(kana_text)
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    
                    x_pos = base_x
                    if direction == 'left':
                        x_pos -= width + random.randint(-3, 1)
                    else:
                        x_pos += random.randint(-1, 3)
                    
                    if is_down:
                        y_pos = center_y + height // 2 + random.randint(0, 3)
                    else:
                        y_pos = center_y - height // 2 + random.randint(-3, 0)
                    
                    write.text((x_pos, y_pos), kana_text, (0, 0, 0), font=font)
                    
                    annotations.append({
                        'type': 'kana',
                        'text': kana_text,
                        'bbox': [x_pos, y_pos, x_pos + width, y_pos + height]
                    })
    
    if have_symbol and random.random() < 0.5:
        symbol = load_character('symbols')
        font = kana_fonts[0]
        bbox_sample = font.getbbox('ア')
        kana_width = bbox_sample[2] - bbox_sample[0]
        symbol_x = center_x - kana_width // 2
        symbol_y = center_y
        
        write.text((symbol_x, symbol_y), symbol, (0, 0, 0), font=font)
        
        bbox = font.getbbox(symbol)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        annotations.append({
            'type': 'symbol',
            'text': symbol,
            'bbox': [symbol_x, symbol_y, symbol_x + width, symbol_y + height]
        })

def regular_img(border=True):
    kanji_font, _ = get_fonts()
    
    img = create_aged_paper_background(900, 1200)
    write = ImageDraw.Draw(img)

    all_annotations = []
    
    x_borders = list(range(50, 800, 90))
    y_positions = list(range(80, 1100, 55))
    
    for x_border in x_borders:
        x_pos = x_border + 25
        
        for i, y_pos in enumerate(y_positions):
            write_char = load_character('common_chars')
            write.text((x_pos, y_pos), write_char, (0, 0, 0), font=kanji_font)
            
            char_annotations = []
            center_x = x_pos + 20
            
            add_annotations(center_x, y_pos + 20, 40, write, False, char_annotations, 0.1)
            add_annotations(center_x, y_pos + 47, 40, write, True, char_annotations, 0.4)
            
            all_annotations.extend(char_annotations)
        
        if border:
            write.rectangle((x_border, 50, x_border + 90, 1150), outline=(0, 0, 0), width=2)
    
    return img, all_annotations

IMAGE_DIR = "results/images/"
LABEL_DIR = "results/labels/"

if __name__ == "__main__":
    # 使用 cv2 直接儲存，避免 PIL 轉換
    for i in tqdm(range(10)):
        img, all_annotations = regular_img(border=True)
        # cvimg = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # cv2.imwrite(f"{IMAGE_DIR}border_{i}.jpg", cvimg)
        img.save(f"{IMAGE_DIR}border_{i}.jpg")
        
        with open(f'{LABEL_DIR}border_{i}.json', 'w', encoding='utf-8') as f:
            json.dump(all_annotations, f, ensure_ascii=False, indent=2)

    for i in tqdm(range(10)):
        img, all_annotations = regular_img(border=False)
        img.save(f"{IMAGE_DIR}noborder_{i}.jpg")
        
        # cvimg = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # cv2.imwrite(f"{IMAGE_DIR}noborder_{i}.jpg", cvimg)
        
        with open(f'{LABEL_DIR}noborder_{i}.json', 'w', encoding='utf-8') as f:
            json.dump(all_annotations, f, ensure_ascii=False, indent=2)