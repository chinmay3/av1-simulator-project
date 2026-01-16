const steps = document.querySelectorAll(".step");
const cursor = document.getElementById("cursor");
const stage = document.getElementById("stage");

const compressBtn = document.getElementById("compressBtn");
const transferBtn = document.getElementById("transferBtn");
const decodeBtn = document.getElementById("decodeBtn");

const yuvFile = document.getElementById("yuvFile");
const av1sFile = document.getElementById("av1sFile");
const mp4File = document.getElementById("mp4File");
const player = document.getElementById("player");

const timeline = [
  { time: 0, step: 1 },
  { time: 1400, step: 2 },
  { time: 3100, step: 3 },
  { time: 4700, step: 4 },
];

const totalDuration = 5600;
let running = false;

function setActiveStep(step) {
  steps.forEach((item) => {
    item.classList.toggle("active", item.dataset.step === String(step));
  });
}

function resetUI() {
  yuvFile.classList.remove("ready");
  av1sFile.classList.remove("ready");
  mp4File.classList.remove("ready");
  player.classList.remove("playing");
  cursor.classList.remove("run");
  steps.forEach((item) => item.classList.remove("active"));
}

function runAnimation() {
  if (running) {
    return;
  }
  running = true;
  resetUI();
  void stage.offsetWidth;

  cursor.classList.add("run");
  timeline.forEach((entry) => {
    setTimeout(() => setActiveStep(entry.step), entry.time);
  });

  setTimeout(() => yuvFile.classList.add("ready"), 600);
  setTimeout(() => av1sFile.classList.add("ready"), 2100);
  setTimeout(() => transferBtn.classList.add("pulse"), 2600);
  setTimeout(() => decodeBtn.classList.add("pulse"), 3800);
  setTimeout(() => mp4File.classList.add("ready"), 4600);
  setTimeout(() => player.classList.add("playing"), 4800);

  setTimeout(() => {
    transferBtn.classList.remove("pulse");
    decodeBtn.classList.remove("pulse");
    running = false;
  }, totalDuration + 200);
}

if (compressBtn && transferBtn && decodeBtn) {
  compressBtn.addEventListener("click", runAnimation);
  transferBtn.addEventListener("click", runAnimation);
  decodeBtn.addEventListener("click", runAnimation);
}
