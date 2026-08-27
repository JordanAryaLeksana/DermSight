import io
import threading
from pathlib import Path

from PIL import Image, UnidentifiedImageError


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png"}


class UploadValidationError(ValueError):
    pass


def validate_image(file_storage):
    if file_storage is None or not file_storage.filename:
        raise UploadValidationError("Pilih foto kulit terlebih dahulu.")

    extension = Path(file_storage.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise UploadValidationError("Format foto harus JPG, JPEG, atau PNG.")

    image_bytes = file_storage.read()
    if not image_bytes:
        raise UploadValidationError("File foto kosong. Silakan pilih foto lain.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()
        with Image.open(io.BytesIO(image_bytes)) as image:
            image_format = image.format
            width, height = image.size
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise UploadValidationError("File tidak terbaca sebagai gambar yang aman.") from exc

    if image_format not in ALLOWED_FORMATS:
        raise UploadValidationError("Isi file bukan gambar JPG atau PNG yang didukung.")
    if width < 32 or height < 32:
        raise UploadValidationError("Resolusi foto terlalu kecil. Gunakan minimal 32 × 32 piksel.")

    expected_extensions = {"JPEG": {".jpg", ".jpeg"}, "PNG": {".png"}}
    if extension not in expected_extensions[image_format]:
        raise UploadValidationError("Ekstensi file tidak sesuai dengan isi gambar.")

    file_storage.stream.seek(0)
    return image_bytes, ALLOWED_FORMATS[image_format]


class PredictionService:
    _predictor = None
    _lock = threading.Lock()

    def __init__(self, model_path, class_names_path, config_path):
        self.settings = (model_path, class_names_path, config_path)

    def _get_predictor(self):
        if self.__class__._predictor is None:
            with self.__class__._lock:
                if self.__class__._predictor is None:
                    from api.inference.predictor import SkinDiseasePredictor
                    self.__class__._predictor = SkinDiseasePredictor(*self.settings)
        return self.__class__._predictor

    def predict(self, image_bytes):
        return self._get_predictor().predict(image_bytes)
