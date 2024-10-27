import easyocr
import re
from datetime import datetime


class TimestampExtractorOCR:
    def __init__(self, language='en', gpu=False):
        # Initialize the EasyOCR reader with the specified language (default: English)
        self.reader = easyocr.Reader([language], gpu=gpu)

    def extract_timestamp(self, frame, crop_area=(0, 0, 400, 40)):
        # Open the image and pass it through the OCR model
        cropped_frame = frame[crop_area[1]:crop_area[3], crop_area[0]:crop_area[2]]
        results = self.reader.readtext(cropped_frame, detail=0)
        # Join all detected text lines into a single string
        extracted_text = " ".join(results)

        pattern = r"(\d{4}).(\d{2}).(\d{2}).(\d{2}).(\d{2}).(\d{2})"

        match = re.search(pattern, extracted_text)
        if match:
            # Extract components and format as yyyy-mm-dd hh:mm:ss
            date_str = f"{match.group(1)}-{match.group(2)}-{match.group(3)} {match.group(4)}:{match.group(5)}:{match.group(6)}"

            # Convert to datetime
            timestamp = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            return timestamp

        return None
