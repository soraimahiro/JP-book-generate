from PIL import Image, ImageDraw, ImageFont, ImageFilter
import json
import random
import numpy as np
from tqdm import tqdm
import cv2

all_chars = None
# 快取字型物件，避免重複載入
font_cache = {}

def load_character(type):
    global all_chars
    if all_chars is None:
        with open('common_char.json', 'r', encoding='utf-8') as f:
            all_chars = json.load(f)

    if type not in all_chars:
        raise ValueError(f"Type '{type}' not found in character data.")
    return random.choice(all_chars[type])

def get_fonts(size=40):
    """根據縮放因子創建字型，保持漢字:假名 = 3:1 的比例"""
    global font_cache
    if size in font_cache:
        return font_cache[size]

    # 假名大小約為漢字的 1/3，但有一些變化
    kana_base_size = size // 3
    
    kanji_font = ImageFont.truetype('font/NotoSerifTC-Regular.ttf', size, encoding='utf-8')
    kana_fonts = [
        ImageFont.truetype('font/YujiSyuku-Regular.ttf', kana_base_size, encoding='utf-8'),
        ImageFont.truetype('font/YujiSyuku-Regular.ttf', kana_base_size + 3, encoding='utf-8'),
        ImageFont.truetype('font/YujiSyuku-Regular.ttf', kana_base_size + 5, encoding='utf-8'),
        # ImageFont.truetype('font/KleeOne-SemiBold.ttf', kana_base_size, encoding='utf-8'),
        # ImageFont.truetype('font/KleeOne-SemiBold.ttf', kana_base_size + 3, encoding='utf-8')
    ]
    
    font_cache[size] = (kanji_font, kana_fonts)
    return kanji_font, kana_fonts

def create_aged_paper_background(width, height):
    """創建模擬古書掃描的背景"""
    base_color = random.randint(235, 255)
    # 使用 int16 進行累積計算，避免多次類型轉換和裁剪
    img_array = np.full((height, width, 3), base_color, dtype=np.int16)
    
    # 1. 添加大範圍的色調變化（優化：在低解析度上計算後放大）
    scale = 10
    small_h, small_w = (height + scale - 1) // scale, (width + scale - 1) // scale
    noise_map = np.zeros((small_h, small_w), dtype=np.float32)
    y_indices_small, x_indices_small = np.ogrid[:small_h, :small_w]

    for _ in range(5):
        center_x = random.randint(0, small_w)
        center_y = random.randint(0, small_h)
        radius = random.randint(300 // scale, 800 // scale)
        intensity = random.randint(-20, 10)
        
        distances = np.sqrt((x_indices_small - center_x)**2 + (y_indices_small - center_y)**2)
        mask = np.clip(1 - distances / radius, 0, 1)
        noise_map += mask * intensity
    
    # 放大回原尺寸並疊加
    noise_large = cv2.resize(noise_map, (width, height))
    img_array += noise_large[:, :, np.newaxis].astype(np.int16)
    
    # 2. 添加中等大小的污漬（優化：只計算局部區域）
    for _ in range(random.randint(10, 20)):
        center_x = random.randint(0, width)
        center_y = random.randint(0, height)
        radius = random.randint(50, 300)
        intensity = random.randint(-40, -10)
        
        # 計算邊界框
        y1, y2 = max(0, center_y - radius), min(height, center_y + radius)
        x1, x2 = max(0, center_x - radius), min(width, center_x + radius)
        
        if x2 > x1 and y2 > y1:
            y_idx, x_idx = np.ogrid[y1:y2, x1:x2]
            distances = np.sqrt((x_idx - center_x)**2 + (y_idx - center_y)**2)
            mask = np.clip(1 - distances / radius, 0, 1) ** 2
            
            adjustment = (mask * intensity).astype(np.int16)
            img_array[y1:y2, x1:x2] += adjustment[:, :, np.newaxis]
    
    # 3. 添加細小噪點
    noise = np.random.randint(-15, 15, (height, width, 3), dtype=np.int16)
    img_array += noise
    
    # 4. 添加小黑點
    num_spots = random.randint(50, 200)
    spot_coords = np.random.randint(0, [width, height], size=(num_spots, 2))
    spot_sizes = np.random.randint(1, 4, size=num_spots)
    darkness_values = np.random.randint(10, 151, size=num_spots)
    
    for (x, y), spot_size, darkness in zip(spot_coords, spot_sizes, darkness_values):
        y1, y2 = max(0, y - spot_size), min(height, y + spot_size + 1)
        x1, x2 = max(0, x - spot_size), min(width, x + spot_size + 1)
        # 直接減去，最後統一裁剪
        img_array[y1:y2, x1:x2] -= darkness
    
    # 最後統一裁剪並轉換類型
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    # 使用 cv2 進行高斯模糊
    img_array = cv2.GaussianBlur(img_array, (5, 5), 3)
    
    img = Image.fromarray(img_array, mode='RGB')
    return img

def add_annotations(center_x, center_y, char_size, write, have_symbol, annotations, prob_kana=0.3, kana_fonts=None):
    """在兩個字中間插入假名和符號"""
    if kana_fonts is None:
        return
    
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

                x_pos = base_x
                if direction == 'left':
                    bbox = font.getbbox(kana_text)
                    width = bbox[2] - bbox[0]
                    x_pos -= width
                    x_pos += random.randint(-1, 5)
                else:
                    x_pos += random.randint(-5, 1)
                y_pos = center_y + random.randint(-3, 3)
                
                write.text((x_pos, y_pos), kana_text, (0, 0, 0), font=font)
                
                # 使用 textbbox 獲取實際繪製的邊界框
                actual_bbox = write.textbbox((x_pos, y_pos), kana_text, font=font)
                
                annotations.append({
                    'type': 'kana',
                    'text': kana_text,
                    'bbox': list(actual_bbox)
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
                        x_pos -= width
                        x_pos += random.randint(-1, 5)
                    else:
                        x_pos += random.randint(-5, 1)
                    
                    if is_down:
                        y_pos = center_y + height // 2 + random.randint(0, 3)
                    else:
                        y_pos = center_y - height // 2 + random.randint(-3, 0)
                    
                    write.text((x_pos, y_pos), kana_text, (0, 0, 0), font=font)
                    
                    # 使用 textbbox 獲取實際繪製的邊界框
                    actual_bbox = write.textbbox((x_pos, y_pos), kana_text, font=font)
                    
                    annotations.append({
                        'type': 'kana',
                        'text': kana_text,
                        'bbox': list(actual_bbox)
                    })
    
    if have_symbol and random.random() < 0.5:
        symbol = load_character('symbols')
        font = kana_fonts[0]
        bbox_sample = font.getbbox('ア')
        kana_width = bbox_sample[2] - bbox_sample[0]
        symbol_x = center_x - kana_width // 2
        symbol_y = center_y
        
        write.text((symbol_x, symbol_y), symbol, (0, 0, 0), font=font)
        
        # 使用 textbbox 獲取實際繪製的邊界框
        actual_bbox = write.textbbox((symbol_x, symbol_y), symbol, font=font)
        
        annotations.append({
            'type': 'symbol',
            'text': symbol,
            'bbox': list(actual_bbox)
        })

def apply_text_defects(img):
    """對文字添加掃描缺陷效果：不規則邊緣和白點"""
    img_array = np.array(img)
    
    # 1. 找出文字區域（黑色區域）
    # 將圖像轉為灰度
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # 二值化，找出文字（較暗的區域）
    _, text_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    # 3. 在文字區域添加白色噪點（向量化優化）
    text_coords = np.where(text_mask > 0)
    num_pixels = len(text_coords[0])
    
    if num_pixels > 0:
        # 隨機選擇一些文字像素點變成白色
        num_white_spots = int(num_pixels * random.uniform(0.01, 0.05))
        indices = np.random.choice(num_pixels, num_white_spots, replace=False)
        
        ys = text_coords[0][indices]
        xs = text_coords[1][indices]
        
        # 設置為背景色或更亮的顏色
        brightness = np.random.randint(200, 256, size=num_white_spots, dtype=np.uint8)
        
        # 利用廣播機制一次性賦值
        img_array[ys, xs] = brightness[:, np.newaxis]
    
    return img_array

def regular_img(border=True, img_width=900, img_height=1200, char_size=40, 
                column_spacing=90, row_spacing=55, margin=50):
    """
    生成可配置的古書頁面圖像
    
    參數:
        border: 是否繪製邊框
        img_width: 圖像寬度
        img_height: 圖像高度
        char_size: 基礎漢字大小
        column_spacing: 列間距
        row_spacing: 行間距
        border_width: 邊框寬度
        margin: 邊距
    """
    img = create_aged_paper_background(img_width, img_height)
    write = ImageDraw.Draw(img)

    all_annotations = []
    
    # 計算可容納的列數和行數
    num_columns = (img_width - margin - margin) // column_spacing
    available_height = img_height - margin - margin - 50
    num_rows = available_height // row_spacing
    # print(f"Columns: {num_columns}, Rows: {num_rows}")
    # print(f'img_width: {img_width}, img_height: {img_height}')
    
    margin_left = margin + random.randint(0, img_width - margin - margin - num_columns * column_spacing)
    x_borders = [margin_left + i * column_spacing for i in range(num_columns)]
    y_positions = [margin + 20 + i * row_spacing for i in range(num_rows)]
    
    # 優化：移出迴圈，避免重複載入字型
    kanji_font, kana_fonts = get_fonts(char_size)
    
    for x_border in x_borders:
        # 根據縮放調整字符大小和位置
        x_offset = int((column_spacing - char_size) / 2)
        x_pos = x_border + x_offset
        
        for i, y_pos in enumerate(y_positions):
            write_char = load_character('common_chars')
            write.text((x_pos, y_pos), write_char, (0, 0, 0), font=kanji_font)
            
            char_annotations = []
            center_x = x_pos + char_size // 2
            
            add_annotations(center_x, y_pos + char_size // 2, char_size, write, False, char_annotations, 0.1, kana_fonts)
            add_annotations(center_x, y_pos + char_size + (row_spacing - char_size) // 2, char_size, write, True, char_annotations, 0.4, kana_fonts)
            
            all_annotations.extend(char_annotations)
        
        if border > 0:
            # print(f'available_height: {available_height}, top: {margin_top}, margin_bottom: {margin_top + available_height + 50}')
            write.rectangle((x_border, margin, x_border + column_spacing, img_height - margin), 
                          outline=(0, 0, 0), width=border)
    
    # 應用文字缺陷效果
    img = apply_text_defects(img)
    
    return img, all_annotations

IMAGE_DIR = "results/images/"
LABEL_DIR = "results/labels/"

if __name__ == "__main__":
    for i in tqdm(range(20)):
        border = random.randint(0, 3)
        img_width = random.randint(1000, 2000)
        img_height = random.randint(1000, 2000)
        min_size = min(img_width, img_height)
        char_size = random.randint(int(min_size/30), int(min_size/15))
        column_spacing = random.randint(int(char_size*1.8), int(char_size*2.2))
        row_spacing = random.randint(int(char_size*1.2), int(char_size*1.5))
        margin = random.randint(30, 150)
        
        img, all_annotations = regular_img(
            border=border,
            img_width=img_width,
            img_height=img_height,
            char_size=char_size,
            column_spacing=column_spacing,
            row_spacing=row_spacing,
            margin=margin
        )
        
        cv2.imwrite(f"{IMAGE_DIR}book{i}.jpg", img)
        
        with open(f'{LABEL_DIR}book{i}.json', 'w', encoding='utf-8') as f:
            json.dump(all_annotations, f, ensure_ascii=False, indent=2)

    # img, all_annotations = regular_img(
    #         border=3,
    #         img_width=2000,
    #         img_height=1000,
    #         char_size=60,
    #         column_spacing=120,
    #         row_spacing=75
    #     )
        
    # cv2.imwrite(f"{IMAGE_DIR}sp.jpg", img)
        
    # with open(f'{LABEL_DIR}sp.json', 'w', encoding='utf-8') as f:
    #     json.dump(all_annotations, f, ensure_ascii=False, indent=2)