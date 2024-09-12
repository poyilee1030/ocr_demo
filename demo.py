import cv2
import time
import numpy as np
from PIL import Image
from glob import glob
from densenet.model import predict as keras_densenet


if __name__ == '__main__':

    result_file = open("result.txt", "w", encoding="utf-8")
    img_list = glob('./test_images/*.jpg')

    for i, img_path in enumerate(img_list):
        t = time.time()
        pil_img = Image.open(img_path).convert('L')

        char_list, prob_list = keras_densenet(pil_img)
        text = "".join(char_list)
        conf = ",".join(prob_list)

        print("Mission complete, it took {:.3f}s".format(time.time() - t))
        print(f"Recognition Result: {text}, {conf}")
        line = str(i).zfill(2) + "\t" + text + "\t" + conf + "\n"
        result_file.write(line)

    result_file.close()
