const form = document.getElementById("batch-form");
const fileInput = document.getElementById("batch-file");
const fileName = document.getElementById("file-name");
const dropzone = document.getElementById("dropzone");
const button = document.getElementById("batch-button");
const errorBox = document.getElementById("batch-error");
const resultSection = document.getElementById("batch-result-section");
const rowCount = document.getElementById("row-count");
const downloadLink = document.getElementById("download-link");
const table = document.getElementById("preview-table");
const countLow = document.getElementById("count-low");
const countModerate = document.getElementById("count-moderate");
const countHigh = document.getElementById("count-high");
const meanConfidence = document.getElementById("mean-confidence");

function updateFileLabel() {
    const file = fileInput.files?.[0];
    fileName.textContent = file ? `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB` : "No file selected";
    dropzone.classList.toggle("has-file", Boolean(file));
}

fileInput.addEventListener("change", updateFileLabel);

["dragenter", "dragover"].forEach((name) => {
    dropzone.addEventListener(name, (event) => {
        event.preventDefault();
        dropzone.classList.add("dragging");
    });
});

["dragleave", "drop"].forEach((name) => {
    dropzone.addEventListener(name, (event) => {
        event.preventDefault();
        dropzone.classList.remove("dragging");
    });
});

dropzone.addEventListener("drop", (event) => {
    const files = event.dataTransfer?.files;
    if (files?.length) {
        fileInput.files = files;
        updateFileLabel();
    }
});

function renderPreview(rows) {
    table.innerHTML = "";
    if (!rows?.length) return;

    const columns = Object.keys(rows[0]);
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");
    columns.forEach((column) => {
        const th = document.createElement("th");
        th.textContent = column;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    rows.forEach((row) => {
        const tr = document.createElement("tr");
        columns.forEach((column) => {
            const td = document.createElement("td");
            const value = row[column];
            td.textContent = value === null || value === undefined ? "—" : String(value);
            if (column === "predicted_risk_label") td.dataset.risk = String(value).toLowerCase();
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
}

function showError(message, messages = []) {
    errorBox.hidden = false;
    resultSection.hidden = true;
    errorBox.replaceChildren();

    const heading = document.createElement("strong");
    heading.textContent = message;
    errorBox.appendChild(heading);

    if (messages.length) {
        const list = document.createElement("ul");
        messages.forEach((item) => {
            const li = document.createElement("li");
            li.textContent = item;
            list.appendChild(li);
        });
        errorBox.appendChild(list);
    }
}

function renderSummary(data) {
    const counts = data.risk_counts || {};
    countLow.textContent = counts.Low ?? 0;
    countModerate.textContent = counts.Moderate ?? 0;
    countHigh.textContent = counts.High ?? 0;
    meanConfidence.textContent = typeof data.average_confidence === "number"
        ? `${(data.average_confidence * 100).toFixed(1)}%`
        : "—";
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    errorBox.hidden = true;
    resultSection.hidden = true;

    if (!fileInput.files?.length) {
        showError("Choose a CSV file first.");
        return;
    }

    button.disabled = true;
    button.classList.add("loading");
    button.querySelector("span").textContent = "Processing CSV…";

    try {
        const body = new FormData(form);
        const base = window.location.pathname.replace(/\/$/, "");
        const response = await fetch(`${base}/predict`, { method: "POST", body });
        const data = await response.json();
        if (!response.ok) {
            showError(data.error || "Batch prediction failed.", data.messages || []);
            return;
        }

        rowCount.textContent = data.row_count;
        downloadLink.href = data.download_url || `${base}/download/${data.result_id}`;
        renderSummary(data);
        renderPreview(data.preview);
        resultSection.hidden = false;
        resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
        showError(error.message || "Could not reach the batch prediction service.");
    } finally {
        button.disabled = false;
        button.classList.remove("loading");
        button.querySelector("span").textContent = "Validate & predict";
    }
});
