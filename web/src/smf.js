// 標準MIDIファイル（SMF）を書き出す。フォーマット0（1トラック）と1（パート別トラック）に対応。
//
// tracks: [{ name, channel, program, notes: [{ time, duration, midi, velocity(0〜1) }] }]

const enc = new TextEncoder();

function vlq(n) {
  const out = [n & 0x7f];
  while ((n >>= 7)) out.unshift((n & 0x7f) | 0x80);
  return out;
}
const u32 = (n) => [(n >>> 24) & 255, (n >>> 16) & 255, (n >>> 8) & 255, n & 255];
const u16 = (n) => [(n >>> 8) & 255, n & 255];

function meta(type, data) {
  return [0xff, type, ...vlq(data.length), ...data];
}

// events: [{ tick, order, bytes }] → MTrk チャンク
function chunk(events) {
  events.sort((a, b) => a.tick - b.tick || a.order - b.order);
  const body = [];
  let last = 0;
  for (const e of events) {
    body.push(...vlq(e.tick - last), ...e.bytes);
    last = e.tick;
  }
  body.push(0, ...meta(0x2f, []));
  return [...enc.encode("MTrk"), ...u32(body.length), ...body];
}

function conductor(tempo) {
  const usPerBeat = Math.round(60000000 / tempo);
  return [
    { tick: 0, order: 0, bytes: meta(0x51, [(usPerBeat >> 16) & 255, (usPerBeat >> 8) & 255, usPerBeat & 255]) },
    { tick: 0, order: 0, bytes: meta(0x58, [4, 2, 24, 8]) }, // 4/4拍子
  ];
}

function trackEvents(tr, toTick, withName) {
  const ev = [];
  const ch = tr.channel & 15;
  if (withName && tr.name) ev.push({ tick: 0, order: 0, bytes: meta(0x03, [...enc.encode(tr.name)]) });
  ev.push({ tick: 0, order: 1, bytes: [0xc0 | ch, tr.program & 127] });
  for (const n of tr.notes) {
    const on = toTick(n.time), off = Math.max(on + 1, toTick(n.time + n.duration));
    const vel = Math.min(127, Math.max(1, Math.round(n.velocity * 127)));
    const key = Math.min(127, Math.max(0, Math.round(n.midi)));
    ev.push({ tick: on, order: 3, bytes: [0x90 | ch, key, vel] });
    ev.push({ tick: off, order: 2, bytes: [0x80 | ch, key, 0] }); // 同時刻なら先に消す
  }
  return ev;
}

export function writeSMF(tracks, { format = 1, tempo = 120, ppq = 480 } = {}) {
  const toTick = (sec) => Math.round(sec * (tempo / 60) * ppq);
  let chunks;
  if (format === 0) {
    const ev = conductor(tempo);
    for (const tr of tracks) ev.push(...trackEvents(tr, toTick, false));
    chunks = [chunk(ev)];
  } else {
    chunks = [chunk(conductor(tempo)), ...tracks.map((tr) => chunk(trackEvents(tr, toTick, true)))];
  }
  const header = [...enc.encode("MThd"), ...u32(6), ...u16(format), ...u16(chunks.length), ...u16(ppq)];
  return new Uint8Array([...header, ...chunks.flat()]);
}
