import io

import pytest
from PIL import Image

from web.app import create_app


class FakePredictionService:
    def predict(self, image_bytes):
        assert image_bytes
        return {"predicted_label": "Eczema", "confidence": 0.87}


class FakeLLMService:
    def analyze(self, label, confidence):
        return {"recommendation": "Ringkasan\nHasil AI menunjukkan kemiripan dengan Eczema.\n\nKapan perlu periksa\nHubungi tenaga kesehatan jika memburuk."}


class FakeEmailService:
    def __init__(self): self.messages = []
    def send_aspiration(self, data): self.messages.append(data)


@pytest.fixture
def app():
    app = create_app({"TESTING": True, "SECRET_KEY": "test-secret", "MAX_CONTENT_LENGTH": 2 * 1024 * 1024})
    app.extensions["prediction_service"] = FakePredictionService()
    app.extensions["llm_service"] = FakeLLMService()
    app.extensions["email_service"] = FakeEmailService()
    return app


@pytest.fixture
def client(app): return app.test_client()


@pytest.fixture
def csrf(client):
    client.get("/")
    with client.session_transaction() as flask_session:
        return flask_session["csrf_token"]


@pytest.fixture
def image_file():
    stream = io.BytesIO()
    Image.new("RGB", (128, 128), (190, 130, 110)).save(stream, format="JPEG")
    stream.seek(0)
    return stream
