from typing import Dict, List, Any

from utils.logger import get_logger


logger = get_logger(__name__)


SYSTEM_PROMPT = """
Anda adalah asisten edukasi kesehatan kulit untuk masyarakat umum, termasuk masyarakat di daerah 3T.

Tugas Anda adalah menjelaskan hasil klasifikasi penyakit kulit berbasis AI dengan bahasa Indonesia yang sederhana, jelas, singkat, dan mudah dipahami.

Aturan:
- Jangan memberikan diagnosis medis pasti.
- Selalu jelaskan bahwa hasil berasal dari model klasifikasi gambar berbasis AI.
- Gunakan hanya data dari knowledge base yang diberikan.
- Jangan menambahkan informasi medis di luar knowledge base.
- Jika informasi tertentu tidak tersedia di knowledge base, tulis bahwa informasi tersebut tidak tersedia.
- Jangan merekomendasikan obat resep.
- Jangan menyebut nama obat.
- Jangan menyuruh pengguna membeli obat tertentu.
- Jangan memberi instruksi tindakan medis yang berisiko.
- Sarankan pengguna memeriksakan diri ke puskesmas, pustu, klinik, posyandu, tenaga kesehatan, dokter, atau dokter spesialis kulit jika tersedia.
- Gunakan bahasa sederhana, tidak terlalu teknis, dan cocok untuk pengguna non-medis.
- Jangan membuat jawaban umum yang tidak informatif.
- Jangan membuat judul baru.
- Jangan mengubah urutan judul.
""".strip()


def _safe_text(value: Any, default: str = "Informasi tidak tersedia dalam knowledge base.") -> str:
    if value is None:
        return default

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "null"}:
        return default

    return text


def _build_confidence_instruction(confidence: float, confidence_percent: float) -> str:
    if confidence < 0.5:
        return (
            f"Tingkat keyakinan AI adalah {confidence_percent}%. "
            "Jelaskan bahwa angka ini masih rendah, sehingga hasil perlu dibaca dengan hati-hati "
            "dan sebaiknya diperiksa oleh tenaga kesehatan."
        )

    if confidence < 0.75:
        return (
            f"Tingkat keyakinan AI adalah {confidence_percent}%. "
            "Jelaskan bahwa angka ini sedang, sehingga hasil masih perlu dikonfirmasi melalui pemeriksaan langsung."
        )

    return (
        f"Tingkat keyakinan AI adalah {confidence_percent}%. "
        "Jelaskan bahwa angka ini cukup tinggi, tetapi tetap bukan diagnosis pasti."
    )


def build_skin_disease_prompt(
    predicted_label: str,
    confidence: float,
    retrieval_result: Dict[str, Any],
) -> List[Dict[str, str]]:
    logger.info(
        "Building skin disease prompt | predicted_label=%s | confidence=%s",
        predicted_label,
        confidence,
    )

    if not isinstance(predicted_label, str) or not predicted_label.strip():
        logger.warning(
            "Invalid predicted_label for prompt builder | value=%s | type=%s",
            predicted_label,
            type(predicted_label),
        )
        raise ValueError("Predicted label must be a non-empty string.")

    if not isinstance(confidence, (float, int)):
        logger.warning(
            "Invalid confidence type for prompt builder | value=%s | type=%s",
            confidence,
            type(confidence),
        )
        raise ValueError("Confidence must be a number (float or int).")

    if confidence < 0 or confidence > 1:
        logger.warning(
            "Confidence out of range for prompt builder | confidence=%s",
            confidence,
        )
        raise ValueError("confidence must be between 0 and 1")

    if not isinstance(retrieval_result, dict):
        logger.warning(
            "Invalid retrieval_result type for prompt builder | type=%s",
            type(retrieval_result),
        )
        raise TypeError("retrieval_result must be a dictionary")

    matched = retrieval_result.get("matched", False)
    match_type = retrieval_result.get("match_type")
    matched_label = retrieval_result.get("matched_label")
    disease = retrieval_result.get("disease")

    logger.info(
        "Retrieval result received for prompt | matched=%s | match_type=%s | matched_label=%s | disease=%s",
        matched,
        match_type,
        matched_label,
        disease,
    )

    if matched:
        disease_name = _safe_text(disease, "Kondisi kulit tidak diketahui")
        matched_label_text = _safe_text(matched_label, "Unknown")
        match_type_text = _safe_text(match_type, "unknown")

        simple_explanation = _safe_text(
            retrieval_result.get("simple_explanation")
        )
        common_signs = _safe_text(
            retrieval_result.get("common_signs")
        )
        safe_actions = _safe_text(
            retrieval_result.get("safe_actions")
        )
        avoid = _safe_text(
            retrieval_result.get("avoid")
        )
        when_to_seek_help = _safe_text(
            retrieval_result.get("when_to_seek_help")
        )
    else:
        logger.warning(
            "No matched retrieval context found. Prompt will use limited information | predicted_label=%s",
            predicted_label,
        )

        disease_name = "Kondisi kulit tidak diketahui"
        matched_label_text = "None"
        match_type_text = "none"

        simple_explanation = "Informasi penjelasan sederhana tidak tersedia dalam knowledge base."
        common_signs = "Informasi tanda yang bisa diperhatikan tidak tersedia dalam knowledge base."
        safe_actions = (
            "Amati perubahan pada kulit; Jaga kebersihan area kulit; "
            "Hindari menggaruk atau memencet; Periksa ke tenaga kesehatan jika keluhan memburuk"
        )
        avoid = (
            "Jangan memakai obat keras tanpa arahan tenaga kesehatan; "
            "Jangan menyimpulkan diagnosis hanya dari AI"
        )
        when_to_seek_help = (
            "Periksa ke tenaga kesehatan jika keluhan memburuk, menyebar, nyeri, "
            "bernanah, berdarah, demam, sering kambuh, atau mengganggu aktivitas."
        )

    confidence = float(confidence)
    confidence_percent = round(confidence * 100, 2)
    confidence_instruction = _build_confidence_instruction(
        confidence=confidence,
        confidence_percent=confidence_percent,
    )

    logger.debug(
        "Prompt confidence converted | confidence=%s | confidence_percent=%s",
        confidence,
        confidence_percent,
    )

    user_prompt = f"""
DATA HASIL AI:
- Label prediksi model: {predicted_label.strip()}
- Label knowledge base yang cocok: {matched_label_text}
- Nama penyakit/kondisi: {disease_name}
- Tingkat keyakinan AI: {confidence_percent}%
- Jenis kecocokan: {match_type_text}

DATA KNOWLEDGE BASE TERSTRUKTUR:
Nama kondisi:
{disease_name}

Penjelasan sederhana:
{simple_explanation}

Tanda yang bisa diperhatikan:
{common_signs}

Yang dapat dilakukan dengan aman:
{safe_actions}

Yang sebaiknya dihindari:
{avoid}

Kapan perlu periksa:
{when_to_seek_help}

TARGET PEMBACA:
Masyarakat umum, termasuk warga daerah 3T, yang mungkin tidak terbiasa dengan istilah medis dan mungkin memiliki akses terbatas ke dokter spesialis.

ATURAN WAJIB:
- Gunakan hanya DATA KNOWLEDGE BASE TERSTRUKTUR.
- Jangan menambahkan informasi medis di luar data tersebut.
- Jika nama penyakit/kondisi tersedia, wajib sebutkan nama tersebut.
- Jangan memberi diagnosis pasti.
- Jangan menyarankan obat resep.
- Jangan menyebut nama obat.
- Jangan menyuruh pengguna mengobati sendiri.
- Jangan menulis kalimat generik seperti:
  "masalah kulit yang umum terjadi",
  "kondisi ini memerlukan perhatian",
  "penjelasan bergantung pada konteks",
  atau kalimat umum lain yang tidak memberi informasi nyata.
- Ubah item yang dipisahkan tanda titik koma menjadi daftar poin.
- Jangan membuat judul baru.
- Jangan mengubah urutan judul.
- Jawaban harus pendek, jelas, menenangkan, dan cocok untuk warga 3T.

FORMAT JAWABAN WAJIB:

Nama kondisi:
Tulis nama kondisi: {disease_name}

Tingkat keyakinan AI:
{confidence_instruction}

Penjelasan sederhana:
Tulis ulang penjelasan sederhana dari knowledge base dengan bahasa awam. Jangan menambahkan informasi baru.

Tanda yang bisa diperhatikan:
- Tulis tanda dari knowledge base sebagai daftar poin.
- Jika informasi tidak tersedia, tulis: Informasi tanda yang bisa diperhatikan tidak tersedia dalam knowledge base.

Yang dapat dilakukan dengan aman:
- Tulis saran aman dari knowledge base sebagai daftar poin.
- Pastikan ada arahan realistis untuk warga 3T seperti puskesmas, pustu, klinik, posyandu, atau tenaga kesehatan terdekat jika tersedia.

Yang sebaiknya dihindari:
- Tulis hal yang perlu dihindari dari knowledge base sebagai daftar poin.
- Jangan menambahkan obat atau tindakan medis berisiko.

Kapan perlu periksa:
Tulis kapan pengguna perlu memeriksakan diri berdasarkan knowledge base.

Catatan penting:
Tulis bahwa hasil ini berasal dari model klasifikasi gambar berbasis AI, bukan diagnosis dokter. Pemeriksaan langsung oleh tenaga kesehatan tetap diperlukan.
""".strip()

    logger.info(
        "Skin disease prompt built successfully | disease=%s | match_type=%s",
        disease_name,
        match_type_text,
    )

    logger.debug(
        "Prompt message details | system_prompt_length=%s | user_prompt_length=%s",
        len(SYSTEM_PROMPT),
        len(user_prompt),
    )

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]