const progressBar = document.getElementById("progressBar");
const progressLabel = document.getElementById("progressLabel");
const av1sFile = document.getElementById("av1sFile");
const mp4File = document.getElementById("mp4File");
const nextBtn = document.getElementById("nextBtn");
const player = document.getElementById("player");

function setProgress(value) {
  if (!progressBar) {
    return;
  }
  progressBar.style.width = `${value}%`;
}

function setLabel(text) {
  if (progressLabel) {
    progressLabel.textContent = text;
  }
}

function showFile(fileEl, sizeEl, sizeText) {
  if (!fileEl) {
    return;
  }
  fileEl.classList.remove("hidden");
  if (sizeEl && sizeText) {
    sizeEl.textContent = sizeText;
  }
}

function pipelineStart(job) {
  const startEndpoint = job === "compress" ? "/api/compress" : "/api/decode";
  fetch(startEndpoint, { method: "POST" }).then(() => {
    pollStatus(job);
  });
}

function pollStatus(job) {
  fetch(`/api/status/${job}`)
    .then((res) => res.json())
    .then((data) => {
      if (data.error) {
        setLabel(`Error: ${data.error}`);
        return;
      }
      setProgress(data.progress);
      if (data.state === "running") {
        setLabel(job === "compress" ? "Compressing..." : "Decoding...");
        setTimeout(() => pollStatus(job), 500);
        return;
      }
      if (data.state === "done") {
        setLabel("Done");
        setProgress(100);
        if (job === "compress") {
          showFile(av1sFile, document.getElementById("av1sSize"), data.output);
          if (nextBtn) {
            nextBtn.classList.remove("hidden");
            nextBtn.addEventListener("click", () => {
              window.location.href = "/transfer";
            });
          }
        } else {
          showFile(mp4File, document.getElementById("mp4Size"), data.output);
          if (player) {
            player.classList.remove("hidden");
          }
        }
        return;
      }
      setLabel("Queued...");
      setTimeout(() => pollStatus(job), 500);
    })
    .catch(() => {
      setLabel("Waiting for server...");
      setTimeout(() => pollStatus(job), 1000);
    });
}

window.pipelineStart = pipelineStart;
