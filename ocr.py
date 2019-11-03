try:
    import tesserocr
except ImportError:
    tesserocr = None

try:
    import pytesseract
except ImportError:
    pytesseract = None


class PyTesseractApi:
    def __init__(self, config):
        self._config = config

    def image_to_string(self, image):
        return pytesseract.image_to_string(image, config=self._config)


class TesserocrApi:
    def __init__(self, ocr_api):
        self._ocr_api = ocr_api

    def image_to_string(self, image):
        self._ocr_api.SetImage(image)
        return self._ocr_api.GetUTF8Text().strip()


def get_ocr_api(*, psm, char_whitelist):
    """
    Configure an OCR API.

    Prefers to use `tesserocr` for its faster performance, but falls back on `pytesseract` if
    `tesserocr` configuration fails.
    """
    if tesserocr:
        try:
            ocr_api = tesserocr.PyTessBaseAPI(psm=psm)
            ocr_api.SetVariable("tessedit_char_whitelist", char_whitelist)
            return TesserocrApi(ocr_api)
        except Exception as exc:
            print("Error with tesserocr api:", exc)
            print("trying to fall back to pytesseract..")

    if not pytesseract:
        raise RuntimeError(
            "Both tesserocr and pytesseract libraries failed. "
            "Make sure at least one is installed and working."
        )

    config = f"--psm {psm} -c tessedit_char_whitelist={char_whitelist}"
    return PyTesseractApi(config)
