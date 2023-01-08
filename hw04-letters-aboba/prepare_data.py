import cv2
import os
import numpy as np
HEIGHT = 80
WEIGHT = 80
data_folder = 'data'


def main(file_name):
    image = cv2.imread(file_name)
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    letter_number = 0
    centers = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = centers[0] if len(centers) == 2 else centers[1]
    ws, hs = [], []
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    for c in centers:
        x, y, w, h = cv2.boundingRect(c)
        center_x = x + w // 2
        center_y = y + h // 2
        left_x, left_y = center_x - WEIGHT // 2, center_y - HEIGHT // 2
        right_x, right_y = center_x + WEIGHT // 2, center_y + HEIGHT // 2
        left_x, left_y = max(left_x, 0), max(left_y, 0)
        right_x, right_y = min(right_x, original.shape[1]), min(right_y, original.shape[0])
        cv2.rectangle(image, (left_x, left_y), (right_x, right_y), (36, 255, 12), 2)
        letter = original[left_y:right_y, left_x:right_x]
        cv2.imwrite(os.path.join(data_folder, f'letter_{center_x}_{center_y}.png'), letter)
        letter_number += 1
        ws.append(w)
        hs.append(h)

    ws, hs = np.array(ws), np.array(hs)
    print(f'Letters number {letter_number}')
    print(f'width mean:{ws.mean()}, max:{ws.max()}, height mean:{hs.mean()}, max:{hs.max()}')


if __name__ == "__main__":
    main('letters_short.png')
