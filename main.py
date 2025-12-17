from PIL import Image, ImageDraw, ImageFont, ImageFilter
import json
import random
import numpy as np
from tqdm import tqdm
import cv2

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
    base_color = random.randint(235, 255)
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
    num_spots = random.randint(50, 200)
    for _ in range(num_spots):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        spot_size = random.randint(1, 3)
        darkness = random.randint(10, 150)
        
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

kana_font = [ImageFont.truetype('font/YujiSyuku-Regular.ttf', 16, encoding='utf-8'),
                ImageFont.truetype('font/YujiSyuku-Regular.ttf', 16, encoding='utf-8'),
                ImageFont.truetype('font/KleeOne-SemiBold.ttf', 18, encoding='utf-8'),
                ImageFont.truetype('font/KleeOne-SemiBold.ttf', 18, encoding='utf-8')]
def add_annotations(center_x, center_y, char_size, write, have_symbol, annotations, prob_kana=0.3):
    """在兩個字中間插入假名和符號"""
    
    # 左右兩個方向
    directions = {
        'left': center_x - char_size // 2,   # 左邊假名的 x 位置
        'right': center_x + char_size // 2   # 右邊假名的 x 位置
    }
    
    for direction, base_x in directions.items():
        if random.random() < prob_kana:
            # 20% 機率插入兩個假名，否則一個
            num_kana = 2 if random.random() < 0.2 else 1
            
            
            if num_kana == 1:
                font = random.choice(kana_font)
                # 單個假名，中間對齊 y
                kana_text = load_character('kana')
                
                # 計算 bounding box
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
                # text up
                kana_text1 = load_character('kana')
                
                font = random.choice(kana_font)
                bbox1 = font.getbbox(kana_text1)
                width1 = bbox1[2] - bbox1[0]
                height1 = bbox1[3] - bbox1[1]
                
                x_pos1 = base_x
                if direction == 'left':
                    x_pos1 -= width1
                    x_pos1 += random.randint(-3, 1)
                else:
                    x_pos1 += random.randint(-1, 3)

                y_pos1 = center_y - height1 // 2 + random.randint(-3, 0)

                write.text((x_pos1, y_pos1), kana_text1, (0, 0, 0), font=font)
                
                
                annotations.append({
                    'type': 'kana',
                    'text': kana_text1,
                    'bbox': [x_pos1, y_pos1, x_pos1 + width1, y_pos1 + height1]
                })

                # text down
                kana_text2 = load_character('kana')

                font = random.choice(kana_font)
                bbox2 = font.getbbox(kana_text2)
                width2 = bbox2[2] - bbox2[0]
                height2 = bbox2[3] - bbox2[1]
                
                x_pos2 = base_x
                if direction == 'left':
                    x_pos2 -= width2
                    x_pos2 += random.randint(-3, 1)
                else:
                    x_pos2 += random.randint(-1, 3)

                y_pos2 = center_y + height2 // 2 + random.randint(0, 3)
                
                write.text((x_pos2, y_pos2), kana_text2, (0, 0, 0), font=font)
                
                annotations.append({
                    'type': 'kana',
                    'text': kana_text2,
                    'bbox': [x_pos2, y_pos2, x_pos2 + width2, y_pos2 + height2]
                })
    
    # 中間位置插入符號 (50% 機率)
    if have_symbol and random.random() < 0.5:
        symbol = load_character('symbols')
        
        font = kana_font[0]
        bbox_sample = font.getbbox('ア')
        kana_width = bbox_sample[2] - bbox_sample[0]
        symbol_x = center_x - kana_width // 2
        symbol_y = center_y
        
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

def regular_img(border=True):
    # 字型設定
    kanji_font = ImageFont.truetype('font/NotoSerifTC-Regular.ttf', 40, encoding='utf-8')
    
    # 建立圖片（垂直排列）- 使用古書背景
    img = create_aged_paper_background(900, 1200)
    write = ImageDraw.Draw(img)

    all_annotations = []
    
    # 繪製垂直排列的漢字
    x_borders = list(range(50, 800, 90))
    y_positions = list(range(80, 1100, 55))
    
    for x_border in x_borders:
        x_pos = x_border + 25
        
        for i, y_pos in enumerate(y_positions):
            write_char = load_character('common_chars')
            
            # 繪製漢字
            write.text((x_pos, y_pos), write_char, (0, 0, 0), font=kanji_font)
            
            # print(f"{write_char} at ({x_pos}, {y_pos})")
            
            char_annotations = []
            # 在兩個字中間插入假名和符號（除了最後一個字）
            center_x = x_pos + 20  # 字的中心 x
            
            add_annotations(center_x, y_pos + 20, 40, write, False, char_annotations, 0.1)
            add_annotations(center_x, y_pos + 47, 40, write, True, char_annotations, 0.4)
            
            # for ann in char_annotations:
            #     print(f"\t- {ann['text']} at {ann['bbox']}")
            all_annotations = all_annotations + char_annotations
        
        if border:
            write.rectangle((x_border, 50, x_border + 90, 1150), outline=(0, 0, 0), width=2)
    
    return img, all_annotations

IMAGE_DIR = "results/images/"
LABEL_DIR = "results/labels/"

if __name__ == "__main__":
    for i in tqdm(range(10)):
        img, all_annotations = regular_img(border=True)
        cvimg = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{IMAGE_DIR}border_{i}.jpg", cvimg)
        
        with open(f'{LABEL_DIR}border_{i}.json', 'w', encoding='utf-8') as f:
            json.dump(all_annotations, f, ensure_ascii=False, indent=2)

    for i in tqdm(range(10)):
        img, all_annotations = regular_img(border=False)
        
        # 儲存圖片
        cvimg = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{IMAGE_DIR}noborder_{i}.jpg", cvimg)
        
        with open(f'{LABEL_DIR}noborder_{i}.json', 'w', encoding='utf-8') as f:
            json.dump(all_annotations, f, ensure_ascii=False, indent=2)
    
        # print(f"\nsaved new.jpg")
        # print(f"Total annotations: {len(all_annotations)}")