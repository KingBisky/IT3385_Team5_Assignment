(function () {
  const form = document.getElementById("predict-form");
  const stamp = document.getElementById("receipt-stamp");
  const amountEl = document.getElementById("receipt-amount");
  const noteEl = document.getElementById("receipt-note");
  const submitBtn = form.querySelector('button[type="submit"]');

  const PREDICT_URL = window.location.pathname.replace(/\/$/, "") + "/predict";
  let debounceTimer = null;
  let lastValue = null;

  function collectPayload() {
    const payload = {};
    form.querySelectorAll("[data-field]").forEach((el) => {
      payload[el.dataset.field] = el.value;
    });
    return payload;
  }

  function animateAmount(target) {
    const start = lastValue ?? target;
    const duration = 400;
    const startTime = performance.now();

    function tick(now) {
      const progress = Math.min(1, (now - startTime) / duration);
      const eased = 1 - Math.pow(1 - progress, 3);
      const value = start + (target - start) * eased;
      amountEl.textContent = "$" + Math.round(value).toLocaleString();
      if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
    lastValue = target;
  }

  function setStamp(state, text) {
    stamp.classList.remove("is-ready", "is-error");
    if (state) stamp.classList.add(state);
    stamp.textContent = text;
  }

  async function runPrediction() {
    setStamp(null, "APPRAISING…");
    submitBtn.disabled = true;

    try {
      const res = await fetch(PREDICT_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(collectPayload()),
      });
      const data = await res.json();

      if (!res.ok) {
        setStamp("is-error", "ERROR");
        amountEl.textContent = "$—";
        noteEl.textContent = data.error || "Something went wrong. Check the model file is in place.";
        return;
      }

      setStamp("is-ready", "APPRAISED");
      animateAmount(data.prediction);
      noteEl.textContent = "Adjust any field and the desk will re-appraise automatically.";
    } catch (err) {
      setStamp("is-error", "ERROR");
      noteEl.textContent = "Could not reach the server. Is the Flask app running?";
    } finally {
      submitBtn.disabled = false;
    }
  }

  function scheduleAutoPredict() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(runPrediction, 450);
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    clearTimeout(debounceTimer);
    runPrediction();
  });

  form.querySelectorAll("[data-field]").forEach((el) => {
    el.addEventListener("input", scheduleAutoPredict);
    el.addEventListener("change", scheduleAutoPredict);
  });

  // Run once on load so the desk isn't blank before the first edit.
  runPrediction();
})();
