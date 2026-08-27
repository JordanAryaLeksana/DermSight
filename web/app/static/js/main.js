(() => {
  "use strict";
  const input = document.querySelector("[data-image-input]");
  const previewBox = document.querySelector("[data-preview-box]");
  const preview = document.querySelector("[data-image-preview]");
  const fileName = document.querySelector("[data-file-name]");
  let previewUrl;
  if (input && previewBox && preview) {
    input.addEventListener("change", () => {
      const file = input.files && input.files[0];
      if (!file) { previewBox.hidden = true; return; }
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      previewUrl = URL.createObjectURL(file);
      preview.src = previewUrl;
      fileName.textContent = file.name;
      previewBox.hidden = false;
    });
  }

  const form = document.querySelector("[data-detection-form]");
  if (form) {
    form.addEventListener("submit", () => {
      const button = form.querySelector("[data-submit-button]");
      const label = form.querySelector("[data-button-label]");
      const note = form.querySelector("[data-loading-note]");
      button.disabled = true;
      label.textContent = "Sedang Menganalisis…";
      note.hidden = false;
    });
  }

  const overlay = document.querySelector("[data-tour-overlay]");
  const openButtons = document.querySelectorAll("[data-open-tutorial]");
  if (!overlay || !openButtons.length) return;
  openButtons.forEach((button) => { button.hidden = false; });
  const steps = [
    { target: "upload", title: "Unggah Foto", copy: "Pilih foto kulit yang terlihat jelas dan memiliki pencahayaan yang cukup." },
    { target: "detect", title: "Mulai Deteksi", copy: "Setelah foto dipilih, tekan tombol ini untuk memulai analisis." },
    { target: "result", title: "Lihat Hasil", copy: "Hasil deteksi dan penjelasannya akan ditampilkan di bagian ini." }
  ];
  let index = 0;
  let highlighted;
  let returnFocus;
  let renderTimer;
  const card = overlay.querySelector(".tour-card");
  const count = overlay.querySelector("[data-tour-count]");
  const title = overlay.querySelector("[data-tour-title]");
  const copy = overlay.querySelector("[data-tour-copy]");
  const back = overlay.querySelector("[data-tour-back]");
  const next = overlay.querySelector("[data-tour-next]");

  function positionCard(target) {
    if (!target) return;
    const gap = 20;
    const rect = target.getBoundingClientRect();
    const cardRect = card.getBoundingClientRect();
    const roomBelow = window.innerHeight - rect.bottom;
    const top = roomBelow >= cardRect.height + gap
      ? rect.bottom + gap
      : Math.max(16, rect.top - cardRect.height - gap);
    const left = Math.min(
      window.innerWidth - cardRect.width - 16,
      Math.max(16, rect.left + (rect.width - cardRect.width) / 2)
    );
    card.style.left = `${left}px`;
    card.style.top = `${top}px`;
  }
  function renderStep() {
    window.clearTimeout(renderTimer);
    if (highlighted) highlighted.classList.remove("tour-highlight");
    const step = steps[index];
    highlighted = document.querySelector(`[data-tour-target="${step.target}"]`);
    count.textContent = `${index + 1} dari ${steps.length}`;
    title.textContent = step.title;
    copy.textContent = step.copy;
    back.hidden = index === 0;
    next.textContent = index === steps.length - 1 ? "Selesai" : "Berikutnya";
    if (highlighted) {
      highlighted.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" });
      const target = highlighted;
      renderTimer = window.setTimeout(() => {
        if (target !== highlighted || overlay.hidden) return;
        highlighted.classList.add("tour-highlight");
        positionCard(highlighted);
      }, window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 380);
    }
  }
  function openTour(event) { returnFocus = event && event.currentTarget; index = 0; overlay.hidden = false; renderStep(); card.focus(); }
  function closeTour() {
    window.clearTimeout(renderTimer);
    if (highlighted) highlighted.classList.remove("tour-highlight");
    overlay.hidden = true;
    card.removeAttribute("style");
    try { localStorage.setItem("dermsightTutorialSeen", "1"); } catch (_error) { /* app remains usable */ }
    if (returnFocus) returnFocus.focus();
  }
  openButtons.forEach((button) => button.addEventListener("click", openTour));
  overlay.querySelector("[data-tour-skip]").addEventListener("click", closeTour);
  back.addEventListener("click", () => { if (index > 0) { index -= 1; renderStep(); } });
  next.addEventListener("click", () => { if (index < steps.length - 1) { index += 1; renderStep(); } else closeTour(); });
  window.addEventListener("resize", () => { if (!overlay.hidden && highlighted) positionCard(highlighted); });
  document.addEventListener("keydown", (event) => { if (!overlay.hidden && event.key === "Escape") closeTour(); });
  try { if (!localStorage.getItem("dermsightTutorialSeen")) openTour(); } catch (_error) { /* app remains usable */ }
})();
