// ブラウザ版の画面処理。変換も解析もすべてこのブラウザの中で行う。
import * as Tone from "tone";
import { BasicPitch, outputToNotesPoly, addPitchBendsToNoteEvents, noteFramesToTime } from "@spotify/basic-pitch";
import { analyze, SR as ANALYZE_SR } from "./analyze.js";
import { writeSMF } from "./smf.js";

const $ = (id) => document.getElementById(id);
const BP_SR = 22050;     // basic-pitch の入力サンプリングレート
const BP_FPS = BP_SR / 256;
let selectedFile = null, midiUrl = null, notes = [], duration = 0;
let basicPitch = null, synth = null, playing = false;

// ---------- 入力 ----------
const drop = $("drop");
drop.onclick = () => $("file").click();
$("file").onchange = (e) => pick(e.target.files[0]);
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); pick(e.dataTransfer.files[0]); };
function pick(f) { if (!f) return; selectedFile = f; $("fileName").textContent = "選択中: " + f.name; }
["onset", "frame"].forEach((id) => $(id).oninput = () => $(id + "V").textContent = Number($(id).value).toFixed(2));

function setStatus(html, err = false) { $("status").innerHTML = html; $("status").className = "status" + (err ? " err" : ""); }
function setProgress(p) {
  $("progress").style.display = p == null ? "none" : "block";
  $("progress").firstElementChild.style.width = `${Math.round((p || 0) * 100)}%`;
}
function busy(on) { $("go").disabled = $("analyze").disabled = on; }

// 音声ファイルを、指定したサンプリングレートのモノラルに変換する
async function decodeMono(file, rate) {
  const buf = await file.arrayBuffer();
  let audio;
  try {
    audio = await new OfflineAudioContext(1, 1, rate).decodeAudioData(buf);
  } catch {
    throw new Error("この音声ファイルは読み込めませんでした。MP3 か WAV で試してください。");
  }
  const out = new Float32Array(audio.length);
  for (let c = 0; c < audio.numberOfChannels; c++) {
    const d = audio.getChannelData(c);
    for (let i = 0; i < d.length; i++) out[i] += d[i] / audio.numberOfChannels;
  }
  return out;
}

// ---------- MIDI 変換 ----------
$("go").onclick = async () => {
  if (!selectedFile) return setStatus("音声ファイルを選んでください。", true);
  stop();
  busy(true);
  $("result").style.display = "none";
  try {
    setStatus('<span class="spinner"></span> 音声を読み込み中…');
    const audio = await decodeMono(selectedFile, BP_SR);
    duration = audio.length / BP_SR;

    setStatus('<span class="spinner"></span> AI モデルを準備中…（初回だけ少し時間がかかります）');
    if (!basicPitch) basicPitch = new BasicPitch(new URL("./model/model.json", location.href).href);

    const frames = [], onsets = [], contours = [];
    setProgress(0);
    await basicPitch.evaluateModel(audio, (f, o, c) => { frames.push(...f); onsets.push(...o); contours.push(...c); },
      (p) => { setProgress(p); setStatus(`<span class="spinner"></span> 音を聞き取り中… ${Math.round(p * 100)}%`); });

    const onset = Number($("onset").value), frame = Number($("frame").value);
    const minLen = Math.max(1, Math.round((Number($("minNote").value) || 120) / 1000 * BP_FPS));
    const minF = Number($("minFreq").value) || null, maxF = Number($("maxFreq").value) || null;
    const events = noteFramesToTime(addPitchBendsToNoteEvents(contours,
      outputToNotesPoly(frames, onsets, onset, frame, minLen, true, maxF, minF, true, 11)));
    notes = events.map((n) => ({ time: n.startTimeSeconds, duration: n.durationSeconds, midi: n.pitchMidi, velocity: n.amplitude }))
      .sort((a, b) => a.time - b.time);

    const format = Number(document.querySelector("input[name=smf]:checked").value);
    const tempo = Number($("tempo").value) || 120;
    const bytes = writeSMF([{ name: "Music", channel: 0, program: 0, notes }], { format, tempo });
    const base = selectedFile.name.replace(/\.[^.]+$/, "") || "converted";
    const name = `${base}${format === 0 ? "_smf0" : ""}.mid`;
    if (midiUrl) URL.revokeObjectURL(midiUrl);
    midiUrl = URL.createObjectURL(new Blob([bytes], { type: "audio/midi" }));
    $("download").href = midiUrl;
    $("download").download = name;
    $("download").textContent = `SMF（フォーマット${format}）をダウンロード`;
    $("meta").textContent = `${name} ・ ${notes.length} 音 ・ ${fmt(duration)}`;
    $("result").style.display = "block";
    drawRoll();
    setStatus(notes.length ? "" : "音を検出できませんでした。感度を下げて試してください。", !notes.length);
  } catch (e) {
    console.error(e);
    setStatus(e.message || "変換に失敗しました。", true);
  } finally {
    setProgress(null);
    busy(false);
  }
};

function fmt(sec) { const m = Math.floor(sec / 60), s = Math.round(sec % 60); return `${m}:${String(s).padStart(2, "0")}`; }

// ピアノロール
const PX = 40;
function drawRoll(playhead = null) {
  const c = $("roll"), ctx = c.getContext("2d"), H = 260;
  const W = Math.max(c.parentElement.clientWidth, Math.ceil(duration * PX) + 20);
  const dpr = window.devicePixelRatio || 1;
  if (c.width !== W * dpr) { c.width = W * dpr; c.height = H * dpr; c.style.width = W + "px"; c.style.height = H + "px"; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  if (!notes.length) return;
  let lo = 127, hi = 0;
  for (const n of notes) { lo = Math.min(lo, n.midi); hi = Math.max(hi, n.midi); }
  lo -= 2; hi += 2;
  const rowH = H / (hi - lo + 1), css = getComputedStyle(document.documentElement);
  ctx.fillStyle = css.getPropertyValue("--line");
  for (let p = lo; p <= hi; p++) if ([1, 3, 6, 8, 10].includes(((p % 12) + 12) % 12)) ctx.fillRect(0, H - (p - lo + 1) * rowH, W, rowH);
  ctx.fillStyle = css.getPropertyValue("--note");
  for (const n of notes) {
    ctx.globalAlpha = 0.45 + 0.55 * n.velocity;
    ctx.fillRect(10 + n.time * PX, H - (n.midi - lo + 1) * rowH, Math.max(2, n.duration * PX), Math.max(2, rowH - 1));
  }
  ctx.globalAlpha = 1;
  if (playhead != null) { ctx.fillStyle = css.getPropertyValue("--ink"); ctx.fillRect(10 + playhead * PX, 0, 2, H); }
}

// 試聴
$("play").onclick = async () => {
  if (playing) return stop();
  if (!notes.length) return;
  await Tone.start();
  synth = new Tone.PolySynth(Tone.Synth, { envelope: { attack: 0.005, decay: 0.2, sustain: 0.3, release: 0.4 } }).toDestination();
  synth.volume.value = -12;
  const start = Tone.now() + 0.1;
  for (const n of notes) synth.triggerAttackRelease(Tone.Frequency(n.midi, "midi").toFrequency(), n.duration, start + n.time, n.velocity);
  playing = true;
  $("play").textContent = "■ 停止";
  const wrap = document.querySelector(".rollwrap");
  const tick = () => {
    if (!playing) return;
    const pos = Tone.now() - start;
    if (pos > duration + 0.5) return stop();
    drawRoll(pos);
    const x = 10 + pos * PX;
    if (x > wrap.scrollLeft + wrap.clientWidth - 60 || x < wrap.scrollLeft) wrap.scrollLeft = x - 60;
    requestAnimationFrame(tick);
  };
  tick();
};
function stop() {
  playing = false;
  if (synth) { synth.releaseAll(); synth.dispose(); synth = null; }
  $("play").textContent = "▶ 試聴";
  if (notes.length) drawRoll();
}

// ---------- 音源解析 ----------
const TYPE = {
  psg:       { label: "PSG系音源",       ex: "矩形波＋ノイズ。AY-3-8910、SN76489、ファミコンの矩形波など" },
  fm:        { label: "FM系音源",        ex: "周波数変調。YM2612（メガドライブ）、YM2203/2608（PC-88/98）、OPL など" },
  wavetable: { label: "波形メモリ系音源", ex: "短い波形のくり返し。PCエンジン、コナミSCC、ナムコ、ゲームボーイ波形ch など" },
  pcm:       { label: "PCM系音源",       ex: "録音した波形を再生。スーパーファミコン、GM/MIDI音源（SC-55など）、生演奏・録音 など" },
};

$("analyze").onclick = async () => {
  if (!selectedFile) return setStatus("音声ファイルを選んでください。", true);
  busy(true);
  $("anResult").style.display = "none";
  try {
    setStatus('<span class="spinner"></span> 音声を読み込み中…');
    const y = await decodeMono(selectedFile, ANALYZE_SR);
    setProgress(0);
    const r = await analyze(y, (p) => { setProgress(p); setStatus(`<span class="spinner"></span> 音源を解析中… ${Math.round(p * 100)}%`); });
    r.title = selectedFile.name;
    renderAnalysis(r);
    setStatus("");
  } catch (e) {
    console.error(e);
    setStatus(e.message || "解析に失敗しました。", true);
  } finally {
    setProgress(null);
    busy(false);
  }
};

function renderAnalysis(r) {
  const order = Object.keys(TYPE).sort((a, b) => r.scores[b] - r.scores[a]);
  const best = order[0], second = order[1];
  const pct = (k) => Math.round(r.scores[k] * 100);
  $("anMeta").textContent = `${r.title} ・ ${fmt(r.duration)}`;
  $("anBest").innerHTML = `<span style="color:var(--t-${best})">${TYPE[best].label}</span> の可能性が高い（${pct(best)}%）`;
  $("anFamily").textContent = best === "pcm"
    ? "方式：PCM系（録音した波形を鳴らす方式。GM/MIDI音源や生演奏もここに入ります）"
    : "方式：チップ合成系（音源チップが波形をその場で作って鳴らす方式）";
  $("anMix").textContent = pct(second) >= 20 ? `${TYPE[second].label}（${pct(second)}%）も混ざっている可能性があります` : "";
  $("anBars").innerHTML = order.map((k) => `
    <div class="bar">
      <div class="top"><b>${TYPE[k].label}</b><span>${pct(k)}%</span></div>
      <div class="ex">${TYPE[k].ex}</div>
      <div class="track-bg"><div class="fill" style="width:0;background:var(--t-${k})" data-w="${pct(k)}"></div></div>
    </div>`).join("");
  requestAnimationFrame(() => document.querySelectorAll("#anBars .fill").forEach((f) => f.style.width = f.dataset.w + "%"));
  $("anTimeline").innerHTML = r.timeline.map((s) =>
    `<div title="${fmt(s.start)}〜${fmt(s.end)}：${TYPE[s.type].label}" style="width:${(s.end - s.start) / r.duration * 100}%;background:var(--t-${s.type})"></div>`).join("");
  $("anLegend").innerHTML = Object.keys(TYPE).map((k) => `<span><i class="dot" style="background:var(--t-${k})"></i>${TYPE[k].label}</span>`).join("")
    + `<span>0:00 〜 ${fmt(r.duration)}</span>`;
  $("anEvidence").innerHTML = (r.evidence.length ? r.evidence : ["目立った特徴が見つかりませんでした"]).map((e) => `<li>${e}</li>`).join("");
  $("anResult").style.display = "block";
  $("anResult").scrollIntoView({ behavior: "smooth", block: "start" });
}
