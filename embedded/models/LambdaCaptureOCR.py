from .TimestampExtractorOCR import TimestampExtractorOCR
import copy

class LambdaCaptureOCR:

    def __init__(self, extractor: TimestampExtractorOCR, frame):
        self.extractor = extractor
        self.frame = frame
        
    def run(self):
        return self.extractor.extract_timestamp(self.frame)
        
    def copy(self):
        return LambdaCaptureOCR(copy.copy(self.extractor), self.frame)