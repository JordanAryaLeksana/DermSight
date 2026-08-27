import logging
import secrets

from flask import Blueprint, current_app, render_template, request, session
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from .services.email_service import EmailConfigurationError, EmailService
from .services.llm_service import LLMService
from .services.prediction_service import PredictionService, UploadValidationError, validate_image


bp = Blueprint("main", __name__)
logger = logging.getLogger(__name__)
ASPIRATION_KINDS = ("Saran", "Kritik", "Laporan", "Aspirasi")


def _valid_csrf():
    expected = session.get("csrf_token", "")
    supplied = request.form.get("csrf_token", "")
    return bool(expected and supplied and secrets.compare_digest(expected, supplied))


def _prediction_service():
    return current_app.extensions.setdefault(
        "prediction_service",
        PredictionService(
            current_app.config["MODEL_PATH"],
            current_app.config["CLASS_NAMES_PATH"],
            current_app.config["MODEL_CONFIG_PATH"],
        ),
    )


def _llm_service():
    return current_app.extensions.setdefault(
        "llm_service",
        LLMService(
            current_app.config["DISEASE_LIST_PATH"],
            current_app.config["OLLAMA_BASE_URL"],
            current_app.config["OLLAMA_MODEL"],
            current_app.config["LLM_TIMEOUT"],
        ),
    )


def _serializer():
    return URLSafeTimedSerializer(current_app.secret_key, salt="dermsight-analysis-v1")


def _display_label(label):
    return label.replace("_", " ").replace("Seborrh Keratoses", "Seborrheic Keratoses")


@bp.get("/")
def detection():
    return render_template("detection.html")


@bp.post("/deteksi")
def detect():
    if not _valid_csrf():
        return render_template("detection.html", error="Sesi formulir berakhir. Muat ulang halaman lalu coba lagi."), 400

    try:
        image_bytes, _mime = validate_image(request.files.get("image"))
        prediction = _prediction_service().predict(image_bytes)
        label = str(prediction["predicted_label"])
        confidence = float(prediction["confidence"])
        if not 0 <= confidence <= 1:
            raise ValueError("Invalid prediction confidence")
        token = _serializer().dumps({"label": label, "confidence": confidence})
        result = {
            "label": label,
            "display_label": _display_label(label),
            "confidence": confidence,
            "confidence_percent": round(confidence * 100),
            "analysis_token": token,
        }
        return render_template("detection.html", result=result)
    except UploadValidationError as exc:
        return render_template("detection.html", error=str(exc)), 400
    except (FileNotFoundError, ValueError, KeyError):
        logger.exception("Prediction configuration or response error")
        return render_template(
            "detection.html",
            error="Sistem analisis belum siap. Silakan coba kembali atau hubungi pengelola.",
        ), 503
    except Exception:
        logger.exception("Skin prediction failed")
        return render_template(
            "detection.html", error="Foto belum dapat dianalisis. Silakan coba foto lain."
        ), 500


@bp.get("/analisis/<token>")
def analysis(token):
    try:
        payload = _serializer().loads(
            token, max_age=current_app.config["ANALYSIS_TOKEN_MAX_AGE"]
        )
        label = str(payload["label"])
        confidence = float(payload["confidence"])
        if not label or not 0 <= confidence <= 1:
            raise BadSignature("Invalid payload")
    except SignatureExpired:
        return render_template("errors/analysis_expired.html"), 410
    except (BadSignature, KeyError, TypeError, ValueError):
        return render_template("errors/analysis_expired.html"), 400

    try:
        detail = _llm_service().analyze(label, confidence)
        return render_template(
            "analysis.html",
            label=_display_label(label),
            confidence_percent=round(confidence * 100),
            recommendation=detail.get("recommendation", ""),
        )
    except Exception:
        logger.exception("LLM analysis failed for label=%s", label)
        return render_template(
            "analysis.html",
            label=_display_label(label),
            confidence_percent=round(confidence * 100),
            analysis_error="Analisis lengkap sedang tidak tersedia. Hasil deteksi Anda tetap dapat digunakan sebagai informasi awal.",
        ), 503


@bp.route("/aspirasi", methods=["GET", "POST"])
def aspiration():
    values = {"kind": "Saran", "name": "", "contact": "", "message": ""}
    if request.method == "GET":
        return render_template("aspiration.html", values=values, kinds=ASPIRATION_KINDS)

    values = {key: request.form.get(key, "").strip() for key in values}
    errors = []
    if not _valid_csrf():
        errors.append("Sesi formulir berakhir. Muat ulang halaman lalu coba lagi.")
    if request.form.get("website", ""):
        return render_template("aspiration.html", success=True, values=values, kinds=ASPIRATION_KINDS)
    if values["kind"] not in ASPIRATION_KINDS:
        errors.append("Pilih jenis aspirasi yang tersedia.")
    if len(values["name"]) > 100:
        errors.append("Nama maksimal 100 karakter.")
    if len(values["contact"]) > 150:
        errors.append("Kontak maksimal 150 karakter.")
    if len(values["message"]) < 10:
        errors.append("Pesan perlu berisi sedikitnya 10 karakter.")
    if len(values["message"]) > 5000:
        errors.append("Pesan maksimal 5.000 karakter.")
    if errors:
        return render_template(
            "aspiration.html", errors=errors, values=values, kinds=ASPIRATION_KINDS
        ), 400

    try:
        service = current_app.extensions.get("email_service") or EmailService(current_app.config)
        service.send_aspiration(values)
        return render_template(
            "aspiration.html", success=True, values={**values, "message": ""}, kinds=ASPIRATION_KINDS
        )
    except EmailConfigurationError:
        logger.error("Aspiration email service is not configured")
    except Exception:
        logger.exception("Failed to send aspiration email")

    return render_template(
        "aspiration.html",
        errors=["Aspirasi belum berhasil dikirim. Input Anda tetap tersimpan; silakan coba lagi."],
        values=values,
        kinds=ASPIRATION_KINDS,
    ), 502


@bp.get("/health")
def health():
    return {"status": "ok", "service": "dermsight-web"}
