import io
import re


def test_get_detection_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Deteksi" in response.data
    assert b"data-tour-overlay" in response.data


def test_valid_image_prediction_and_analysis(client, csrf, image_file):
    response = client.post("/deteksi", data={"csrf_token": csrf, "image": (image_file, "kulit.jpg")}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"Eczema" in response.data
    assert b"87%" in response.data
    match = re.search(rb'href="(/analisis/[^"]+)"', response.data)
    assert match
    analysis = client.get(match.group(1).decode().replace("&amp;", "&"))
    assert analysis.status_code == 200
    assert b"Penjelasan edukatif" in analysis.data
    assert b"tenaga kesehatan" in analysis.data


def test_invalid_image_is_rejected(client, csrf):
    response = client.post("/deteksi", data={"csrf_token": csrf, "image": (io.BytesIO(b"not an image"), "kulit.jpg")}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert b"tidak terbaca sebagai gambar" in response.data


def test_missing_image_is_rejected(client, csrf):
    response = client.post("/deteksi", data={"csrf_token": csrf})
    assert response.status_code == 400
    assert b"Pilih foto kulit" in response.data


def test_get_aspiration(client):
    response = client.get("/aspirasi")
    assert response.status_code == 200
    assert b"Sampaikan" in response.data


def test_valid_aspiration_sends_email(app, client, csrf):
    response = client.post("/aspirasi", data={"csrf_token": csrf, "kind": "Saran", "name": "Warga", "contact": "", "message": "Mohon dibuat semakin mudah digunakan."})
    assert response.status_code == 200
    assert b"Aspirasi berhasil dikirim" in response.data
    assert app.extensions["email_service"].messages[0]["kind"] == "Saran"


def test_invalid_aspiration_keeps_input(client, csrf):
    response = client.post("/aspirasi", data={"csrf_token": csrf, "kind": "Saran", "name": "", "contact": "", "message": "pendek"})
    assert response.status_code == 400
    assert b"pendek" in response.data
    assert b"sedikitnya 10 karakter" in response.data


def test_csrf_is_required(client):
    response = client.post("/aspirasi", data={"kind": "Saran", "message": "Pesan cukup panjang"})
    assert response.status_code == 400
    assert b"Sesi formulir berakhir" in response.data


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json["status"] == "ok"


def test_404(client):
    response = client.get("/tidak-ada")
    assert response.status_code == 404
    assert b"Halaman tidak ditemukan" in response.data
