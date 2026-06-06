from typing import Dict, List, Any

from utils.logger import get_logger


logger = get_logger(__name__)

SYSTEM_PROMPT = """
Anda adalah asisten edukasi medis yang berfokus pada informasi penyakit kulit.

Tugas Anda adalah menjelaskan hasil klasifikasi penyakit kulit berbasis AI dengan cara yang aman,
jelas, dan mudah dipahami oleh pasien.

Aturan:
- Jangan memberikan diagnosis medis yang pasti.
- Selalu jelaskan bahwa hasil ini berasal dari model klasifikasi gambar berbasis AI.
- Gunakan hanya konteks penyakit yang diberikan oleh knowledge base.
- Jika konteks tidak tersedia atau informasinya terbatas, jelaskan dengan jelas bahwa informasi yang tersedia terbatas.
- Jangan merekomendasikan obat resep.
- Jangan memberikan diagnosis kondisi darurat.
- Sarankan pengguna untuk berkonsultasi dengan dokter atau dokter spesialis kulit yang berkualifikasi.
- Gunakan bahasa sederhana yang mudah dipahami oleh pengguna non-medis.
- Bersikap tenang, sopan, dan meyakinkan.
""".strip()

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
    context = retrieval_result.get("context")

    logger.info(
        "Retrieval result received for prompt | matched=%s | match_type=%s | matched_label=%s | disease=%s",
        matched,
        match_type,
        matched_label,
        disease,
    )
    
    if matched:
        logger.info(
            "Using matched disease context for prompt | disease=%s | match_type=%s",
            disease,
            match_type,
        )

        disease_name = str(disease).strip() if disease else "Unknown skin condition"
        disease_context = str(context).strip() if context else "No disease context was provided."
        matched_label_text = str(matched_label).strip() if matched_label else "Unknown"
        match_type_text = str(match_type).strip() if match_type else "unknown"
    else:
        logger.warning(
            "No matched retrieval context found. Prompt will use limited information | predicted_label=%s",
            predicted_label,
        )

        disease_name = "Unknown skin condition"
        disease_context = "No reliable disease context was found for this prediction."
        matched_label_text = "None"
        match_type_text = "none"
        
    confidence_percent = round(float(confidence * 100), 2)

    logger.debug(
        "Prompt confidence converted | confidence=%s | confidence_percent=%s",
        confidence,
        confidence_percent,
    )

    user_prompt = f"""
    Model klasifikasi gambar berbasis AI menghasilkan hasil berikut:

    Label prediksi dari model:
    {predicted_label.strip()}

    Label yang cocok di knowledge base:
    {matched_label_text}

    Nama penyakit/kondisi:
    {disease_name}

    Tingkat keyakinan model:
    {confidence_percent}%

    Jenis kecocokan:
    {match_type_text}

    Konteks dari knowledge base:
    {disease_context}

    Tugas:
    Buat penjelasan untuk pengguna dalam Bahasa Indonesia.

    Aturan penting:
    - Gunakan hanya informasi dari konteks knowledge base.
    - Jangan memberikan diagnosis pasti.
    - Jangan merekomendasikan obat resep.
    - Jangan menambahkan informasi medis di luar konteks.
    - Jangan mengubah urutan judul.
    - Jangan menambahkan judul baru.
    - Gunakan bahasa sederhana dan mudah dipahami.
    - Jawaban harus konsisten, rapi, dan tidak terlalu panjang.
    - Setiap bagian maksimal 2 sampai 4 kalimat, kecuali bagian daftar poin.

    Gunakan format persis seperti di bawah ini:

    Kemungkinan kondisi:
    Jelaskan bahwa model AI mendeteksi kemungkinan kondisi {disease_name}. Jelaskan secara singkat bahwa hasil ini berasal dari analisis gambar oleh AI dan bukan diagnosis pasti.

    Tingkat keyakinan AI:
    Jelaskan bahwa tingkat keyakinan model adalah {confidence_percent}%. Terangkan bahwa angka ini menunjukkan keyakinan model terhadap hasil gambar, tetapi belum tentu memastikan kondisi sebenarnya.

    Penjelasan sederhana:
    Jelaskan kondisi ini dengan bahasa awam berdasarkan konteks knowledge base. Buat penjelasan yang mudah dimengerti oleh pengguna non-medis.

    Tanda-tanda umum:
    - Sebutkan tanda atau gejala dari konteks knowledge base.
    - Jangan menambahkan tanda yang tidak disebutkan dalam konteks.
    - Jika konteks tidak menyebutkan tanda/gejala, tulis: Informasi tanda-tanda umum tidak tersedia dalam knowledge base.

    Langkah yang disarankan:
    - Amati perubahan pada area kulit.
    - Jaga kebersihan area kulit.
    - Hindari menggaruk, memencet, atau mengobati sendiri tanpa arahan tenaga kesehatan.
    - Konsultasikan dengan dokter atau dokter spesialis kulit untuk pemeriksaan lebih lanjut.

    Hal yang sebaiknya dihindari:
    - Jangan menyimpulkan diagnosis hanya dari hasil AI.
    - Jangan menggunakan obat resep tanpa arahan dokter.
    - Jangan mengabaikan keluhan jika semakin parah atau tidak membaik.

    Kapan perlu ke dokter:
    Sarankan pengguna berkonsultasi dengan dokter jika keluhan menetap, memburuk, menyebar, terasa nyeri, berdarah, bernanah, sering kambuh, atau mengganggu aktivitas.

    Catatan medis:
    Hasil ini bukan diagnosis medis. Hasil ini berasal dari model klasifikasi gambar berbasis AI dan perlu dikonfirmasi oleh dokter atau tenaga kesehatan yang berkualifikasi melalui pemeriksaan langsung.
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