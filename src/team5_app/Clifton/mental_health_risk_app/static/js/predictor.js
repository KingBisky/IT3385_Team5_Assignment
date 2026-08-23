const form = document.getElementById("risk-form");
const button = document.getElementById("predict-button");
const resetButton = document.getElementById("reset-button");
const placeholder = document.getElementById("result-placeholder");
const content = document.getElementById("result-content");
const errorBox = document.getElementById("result-error");

const riskBadge = document.getElementById("risk-badge");
const riskHeading = document.getElementById("risk-heading");
const riskDescription = document.getElementById("risk-description");
const confidenceBadge = document.getElementById("confidence-badge");

function clearFieldErrors() {
    document.querySelectorAll(".field-error").forEach((el) => (el.textContent = ""));
    document.querySelectorAll(".field-card").forEach((el) => el.classList.remove("invalid"));
}

function payloadFromForm() {
    const data = new FormData(form);
    return Object.fromEntries(data.entries());
}

function setLoading(isLoading) {
    button.disabled = isLoading;
    button.classList.toggle("loading", isLoading);
    button.querySelector("span").textContent = isLoading ? "Running model…" : "Run risk prediction";
}

function probability(label, value) {
    const slug = label.toLowerCase();
    const bar = document.getElementById(`prob-${slug}`);
    const text = document.getElementById(`prob-${slug}-value`);
    if (!bar || !text) return;
    const pct = typeof value === "number" ? Math.max(0, Math.min(100, value * 100)) : 0;
    bar.style.width = `${pct}%`;
    text.textContent = typeof value === "number" ? `${pct.toFixed(1)}%` : "—";
}

function showResult(data) {
    placeholder.hidden = true;
    errorBox.hidden = true;
    content.hidden = false;

    const label = data.label || "Unknown";
    riskBadge.className = `risk-badge risk-${label.toLowerCase()}`;
    riskBadge.textContent = label;
    riskHeading.textContent = `${label} mental-health risk`;
    riskDescription.textContent = data.description || "Prediction complete.";

    if (typeof data.confidence === "number") {
        confidenceBadge.textContent = `Confidence ${(data.confidence * 100).toFixed(1)}%`;
    } else {
        confidenceBadge.textContent = "Confidence unavailable";
    }

    probability("Low", data.probabilities?.Low);
    probability("Moderate", data.probabilities?.Moderate);
    probability("High", data.probabilities?.High);
}

function showError(message) {
    placeholder.hidden = true;
    content.hidden = true;
    errorBox.hidden = false;
    errorBox.replaceChildren();

    const heading = document.createElement("strong");
    heading.textContent = "Prediction could not run.";
    const detail = document.createElement("p");
    detail.textContent = message;
    errorBox.append(heading, detail);
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearFieldErrors();
    setLoading(true);

    try {
        const response = await fetch(`${window.location.pathname.replace(/\/$/, "")}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payloadFromForm()),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Prediction failed.");
        showResult(data);
    } catch (error) {
        showError(error.message || "Could not reach the prediction service.");
    } finally {
        setLoading(false);
    }
});

resetButton.addEventListener("click", () => {
    form.reset();
    clearFieldErrors();
    content.hidden = true;
    errorBox.hidden = true;
    placeholder.hidden = false;
});
