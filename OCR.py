from paddleocr import PaddleOCR, draw_ocr
import cv2
import numpy as np
import re
import json
import textwrap
from matplotlib import pyplot as plt
ocr = PaddleOCR(use_angle_cls=True, lang='en')

img_path = 'test.jpg'
result = ocr.ocr(img_path, cls=True)
image = cv2.imread(img_path)
lines = [line[1][0] for line in result[0]]
print("Hasil:", lines)

parsed_data = []
unknown_data = []
current_transaction = {
    'nominal': "Rp.0",
    'name': "Unknown",
    'nomor_rekening': "Unknown"
}

for line in lines:
    if re.search(r'[+-]Rp[\d\.]+', line):
        if current_transaction['nominal'] != "Rp.0":  
            if current_transaction['name'] == "Unknown":
                unknown_data.append(current_transaction)
                raw_text = line
            else:
                parsed_data.append(current_transaction)
            current_transaction = {
                'nominal': "Rp.0",
                'name': "Unknown",
                'nomor_rekening': "Unknown"
            }
        nominal = re.sub(r'[^\d]', '', line.replace("Rp", ""))[:-2]
        current_transaction['nominal'] = f"Rp.{nominal}"

    elif re.search(r'\d{10,20}', line):
        match = re.search(r'([A-Z]+)(\d+)', line)
        if match:
            name = match.group(1).strip()
            account = match.group(2).strip()
            current_transaction['name'] = name
            current_transaction['nomor_rekening'] = account
            
if current_transaction['nominal'] != "Rp.0":
    if current_transaction['name'] == "Unknown":
        unknown_data.append(current_transaction)
    else:
        parsed_data.append(current_transaction)

final_data = {
    "known_transactions": parsed_data,
    "unknown_transactions": unknown_data
}

boxes = [line[0] for line in result[0]]
texts = [line[1][0] for line in result[0]]
scores = [line[1][1] for line in result[0]]

image_with_boxes = draw_ocr(image, boxes, texts, scores, font_path='simfang.ttf')

panel_width = 500
panel_height = image.shape[0]
panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
panel.fill(0)  
def wrap_text(text, font, max_width):
    wrapped_text = []
    for line in text.splitlines():
        if cv2.getTextSize(line, font, 0.7, 2)[0][0] > max_width:
            wrapped_text.extend(textwrap.wrap(line, width=40)) 
        else:
            wrapped_text.append(line)
    return wrapped_text

y_offset = 50
max_width = panel_width - 20

for transaction in final_data["known_transactions"]:
    text = f"Nominal: {transaction['nominal']}, Name: {transaction['name']}, No Rek: {transaction['nomor_rekening']}"
    wrapped_text = wrap_text(text, cv2.FONT_HERSHEY_SIMPLEX, max_width)
    for line in wrapped_text:
        cv2.putText(panel, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y_offset += 30 

for transaction in final_data["unknown_transactions"]:
    text = f"Unknown -> Nominal: {transaction['nominal']}, No Rek: {transaction['nomor_rekening']} ,Raw Text: {transaction['name']}"
    wrapped_text = wrap_text(text, cv2.FONT_HERSHEY_SIMPLEX, max_width)
    for line in wrapped_text:
        cv2.putText(panel, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y_offset += 30  

if panel.shape[0] != image_with_boxes.shape[0]:
    panel = cv2.resize(panel, (panel_width, image_with_boxes.shape[0]))
combined_image = cv2.hconcat([image_with_boxes, panel])

output_path = 'outputsample.jpg'
cv2.imwrite(output_path, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

json_output_path = 'jsonsample.json'
with open(json_output_path, 'w') as json_file:
    json.dump(final_data, json_file, indent=4)

plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
plt.show()