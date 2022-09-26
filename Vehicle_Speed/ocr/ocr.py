import cv2
import pytesseract
import numpy as np
import glob

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

class PlateReader:

    def tesseract_ocr(self, image, lang="eng1", psm=13, oem=3):
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXY0123456789"
        options = "-l {} --psm {} --oem {} -c tessedit_char_whitelist={}".format(lang, psm, oem, alphanumeric)

        return pytesseract.image_to_string(image, config=options)
