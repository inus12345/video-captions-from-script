#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forced-alignment karaoke captions (global-CTC first, chunked fallback):
- Global CTC alignment across the whole audio eliminates drift.
- Chunked alignment remains as a robust fallback or opt-in mode.
- Punctuation & word-boundary safe mapping (apostrophes/hyphens preserved).
- A/V stream start-time offset correction via ffprobe (so subs start at 0).
- **New:** End-to-end warp to the true end of speech/audio with configurable cap.
- ASS output with per-word \k highlighting + slight linger for readability.

Install:
  pip install torch torchaudio ctc-segmentation pyyaml numpy
(For CPU wheels)  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
System dep: ffmpeg
"""

import os, re, sys, math, shlex, argparse, subprocess
from typing import List, Tuple, Dict
import numpy as np
import yaml
import torch
import torchaudio
from ctc_segmentation import CtcSegmentationParameters, prepare_text, ctc_segmentation

# ---------- text cleaning ----------
_WORD_RE_DISPLAY = re.compile(r"[a-z0-9][a-z0-9'-]*", re.IGNORECASE)
_BRACKET_LINE_RE = re.compile(r"^\s*\[.*?\]\s*$")
_TITLE_HINT_RE   = re.compile(r"(full[- ]length recap|segment\s*\d+|intro|prologue|wrap[- ]?up)", re.I)

def normspace(s:str)->str: 
    return re.sub(r"\s+"," ", s.strip())

def normalize_unicode_punct(s: str) -> str:
    s = s.replace("’","'").replace("‘","'")
    s = s.replace("“",'"').replace("”",'"')
    s = s.replace("—","-").replace("–","-")
    s = s.replace("…","...")
    return s

def load_and_clean_script(path: str, drop_title_lines: bool = True) -> str:
    """
    Keep narration only: drop [Segment ...] headers and (optionally) very-early title-ish lines,
    then normalize punctuation and whitespace.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    kept=[]
    for i, ln in enumerate(lines):
        raw = ln.strip()
        if not raw: 
            continue
        if _BRACKET_LINE_RE.match(raw): 
            continue
        if drop_title_lines and i < 3 and (_TITLE_HINT_RE.search(raw) or raw.startswith("🎬")):
            continue
        if not re.search(r"[A-Za-z]", raw): 
            continue
        kept.append(raw)
    return normalize_unicode_punct(normspace(" ".join(kept)))

def words_for_display(s: str) -> List[str]:
    # Keep "don't", "we're", "rock-n-roll" as single words
    return [m.group(0).lower() for m in _WORD_RE_DISPLAY.finditer(s)]

def normalize_for_labels(text: str, labels: List[str]) -> str:
    """
    Convert text to the label alphabet for ctc_segmentation:
      - Uppercase; spaces -> '|'
      - Keep apostrophe/hyphen only if present in labels
      - Drop other chars not in labels
    """
    text = normalize_unicode_punct(text)
    keep = set(labels)
    have_apos   = ("'" in keep)
    have_hyphen = ("-" in keep)

    t = re.sub(r"[^A-Za-z' -]+", " ", text)
    t = re.sub(r"\s+", " ", t).strip().upper()

    out=[]
    for ch in t:
        if ch == " ": lab = "|"
        elif ch == "'":
            if not have_apos:  continue
            lab = "'"
        elif ch == "-":
            if not have_hyphen: continue
            lab = "-"
        else:
            lab = ch
        if lab in keep:
            out.append(lab)

    s = "".join(out)
    s = re.sub(r"\|{2,}", "|", s)
    s = s.strip("|")
    return s

# ---------- shell / io ----------
def _abspath(p): 
    return os.path.abspath(os.path.expanduser(p))

def _must_exist(label,p):
    if not os.path.isfile(p): 
        raise FileNotFoundError(f"{label} not found: {p}")

def run(cmd:str):
    print(f"\n$ {cmd}")
    p = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout)
    if p.returncode!=0: 
        raise RuntimeError(f"Command failed ({p.returncode}): {cmd}")

# ---------- ffprobe A/V offset ----------
def _ffprobe_start_time(video: str, selector: str) -> float:
    # selector: "v:0" or "a:0"
    cmd = f'ffprobe -v error -select_streams {selector} -show_entries stream=start_time -of default=nw=1:nk=1 "{video}"'
    p = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    s = (p.stdout or "").strip()
    try:
        return float(s)
    except:
        return 0.0

def probe_av_offset_seconds(video: str) -> float:
    """
    Return (video_start - audio_start). Positive: shift subs FORWARD on the timeline.
    """
    v0 = _ffprobe_start_time(video, "v:0")
    a0 = _ffprobe_start_time(video, "a:0")
    off = v0 - a0
    return float(np.clip(off, -5.0, 5.0))

# ---------- audio & emissions ----------
def extract_mono_wav(video, out_wav, sr=16000):
    os.makedirs(os.path.dirname(out_wav) or ".", exist_ok=True)
    run(f'ffmpeg -y -i "{video}" -vn -ac 1 -ar {sr} -f wav "{out_wav}"')
    wav, got_sr = torchaudio.load(out_wav)  # [1, T]
    if got_sr != sr:
        wav = torchaudio.functional.resample(wav, got_sr, sr)
    return wav.squeeze(0), sr

def get_emissions(model, waveform: torch.Tensor, sr: int, device: str):
    wf = waveform.unsqueeze(0).to(device)
    with torch.inference_mode():
        em, *rest = model(wf)  # [1, T, C]
        em = torch.log_softmax(em, dim=-1).squeeze(0).cpu()  # [T, C]
    total_sec = waveform.shape[0] / sr
    frame_dur = total_sec / em.shape[0]
    return em, frame_dur  # [T, C], seconds per frame

# ---------- pause detection (simple energy VAD) ----------
def detect_pauses(wav: torch.Tensor, sr: int, frame_ms=30, hop_ms=15,
                  energy_thresh_db=-35.0, min_pause_s=0.6, max_utter_s=20.0) -> List[Tuple[float,float]]:
    x = wav.numpy()
    frame = int(sr * frame_ms / 1000); hop = int(sr * hop_ms / 1000)
    if frame <= 0 or hop <= 0: 
        return [(0.0, len(x)/sr)]
    frames=[]
    for i in range(0, len(x)-frame+1, hop):
        win = x[i:i+frame]; rms = float(np.sqrt(np.mean(win**2) + 1e-12))
        frames.append(20*np.log10(rms + 1e-12))
    frames = np.array(frames); times = (np.arange(len(frames))*hop)/sr
    speech = frames > energy_thresh_db

    regions=[]; in_sp=False; s0=0.0
    for t, sp in zip(times, speech):
        if sp and not in_sp: 
            in_sp=True; s0=t
        elif not sp and in_sp: 
            in_sp=False; regions.append((s0, t))
    if in_sp: 
        regions.append((s0, times[-1] + frame/sr))
    if not regions: 
        return [(0.0, len(x)/sr)]

    # merge & cap length
    utt=[]; cs,ce = regions[0]
    for s,e in regions[1:]:
        if s - ce < min_pause_s and (e - cs) < max_utter_s: 
            ce=e
        else: 
            utt.append((cs,ce)); cs,ce = s,e
    utt.append((cs,ce))

    out=[]
    for s,e in utt:
        if (e - s) <= max_utter_s: 
            out.append((s,e))
        else:
            n = int(math.ceil((e - s)/max_utter_s))
            for k in range(n):
                ss = s + k*max_utter_s; ee = min(e, ss + max_utter_s)
                out.append((ss, ee))
    return out

# ---------- chunking helpers ----------
def words_to_chunks(script_words: List[str], utter_times: List[Tuple[float,float]]) -> List[List[str]]:
    total_dur = sum(max(0.01, e-s) for s,e in utter_times)
    total_words = len(script_words)
    if total_dur <= 0 or total_words == 0: 
        return [script_words]
    raw_counts = [max(1, int(round(total_words * (e-s)/total_dur))) for (s,e) in utter_times]
    diff = sum(raw_counts) - total_words
    counts = raw_counts[:]
    while diff > 0:
        i = int(np.argmax(counts))
        if counts[i] > 1: counts[i]-=1; diff-=1
        else:
            for j in range(len(counts)):
                if counts[j]>1: counts[j]-=1; diff-=1; break
            else: break
    while diff < 0:
        i = int(np.argmin(counts)); counts[i]+=1; diff+=1
    chunks=[]; idx=0
    for c in counts:
        chunks.append(script_words[idx: idx+c]); idx+=c
    if idx < total_words:
        if chunks: chunks[-1].extend(script_words[idx:])
        else: chunks.append(script_words[idx:])
    return chunks

def autosubchunk(words_chunk: List[str], t0: float, t1: float,
                 max_chars: int = 320, max_sec: float = 14.0) -> List[Tuple[List[str], Tuple[float,float]]]:
    text = " ".join(words_chunk); nchar = len(text); dur = max(0.01, t1 - t0)
    if nchar <= max_chars and dur <= max_sec: 
        return [(words_chunk, (t0, t1))]
    mid = max(1, len(words_chunk)//2)
    return autosubchunk(words_chunk[:mid], t0, t0+dur/2, max_chars, max_sec) + \
           autosubchunk(words_chunk[mid:], t0+dur/2, t1, max_chars, max_sec)

# ---------- CTC alignment wrappers ----------
def _ctc_params(labels, blank_id, frame_dur):
    p = CtcSegmentationParameters()
    p.char_list = labels
    p.blank = blank_id
    p.replace_spaces_with_blanks = False
    p.min_token_prob = 0.0
    p.index_duration = frame_dur
    return p

def align_ctc_slice(em_slice, labels, text_str, blank_id, frame_dur, slice_start_sec: float) -> np.ndarray:
    params = _ctc_params(labels, blank_id, frame_dur)
    gt, _ = prepare_text(params, [text_str])
    timings_rel, _, _ = ctc_segmentation(params, em_slice.numpy(), gt)
    return timings_rel + slice_start_sec

def align_ctc_global(emissions, labels, text_str, blank_id, frame_dur) -> np.ndarray:
    params = _ctc_params(labels, blank_id, frame_dur)
    gt, _ = prepare_text(params, [text_str])
    timings, _, _ = ctc_segmentation(params, emissions.numpy(), gt)
    return timings

# ---------- mapping chars → words robustly ----------
def map_char_timings_to_words(ctc_text: str, char_times: np.ndarray, display_words: List[str]) -> List[Dict]:
    """
    Split ctc_text on '|' (CTC space) to segments; map each segment to one display word.
    Handles mismatched counts via proportional distribution across time.
    """
    if not ctc_text:
        return []

    segments = ctc_text.split("|") if "|" in ctc_text else [ctc_text]
    seg_times = []
    idx = 0
    for seg in segments:
        n = len(seg)
        if n == 0:
            t = char_times[idx-1] if idx > 0 else (char_times[idx] if idx < len(char_times) else 0.0)
            seg_times.append((float(t), float(t + 0.12)))
            continue
        t_start = char_times[idx]
        t_end   = char_times[idx + n - 1] if idx + n - 1 < len(char_times) else t_start + 0.12
        seg_times.append((float(t_start), float(max(t_end, t_start+0.08))))
        idx += n

    words = []
    if len(segments) == len(display_words):
        for w, (ts, te) in zip(display_words, seg_times):
            words.append({"word": w, "start": ts, "end": te})
        return words

    # Proportional mapping: slice total time span into len(display_words) equal time buckets
    total_start = seg_times[0][0]
    total_end   = seg_times[-1][1]
    total_dur   = max(0.001, total_end - total_start)
    ideal = [total_start + i * (total_dur/len(display_words)) for i in range(len(display_words)+1)]

    def seg_for_time(t):
        for i,(s,e) in enumerate(seg_times):
            if t <= e + 1e-6:
                return i
        return len(seg_times)-1

    for i in range(len(display_words)):
        s_t = ideal[i]
        e_t = ideal[i+1]
        si  = seg_for_time(s_t)
        ei  = seg_for_time(e_t)
        s_actual = max(s_t, seg_times[si][0])
        e_actual = min(e_t, seg_times[ei][1])
        if e_actual <= s_actual:
            e_actual = s_actual + 0.08
        words.append({"word": display_words[i], "start": float(s_actual), "end": float(e_actual)})
    return words

# ---------- ASS helpers ----------
def duration_to_k_cs(d: float) -> int: 
    return max(1, int(round(max(0.05, d)*100)))

ASS_HEADER = """[Script Info]
; Generated by auto_karaoke_captions.py
ScriptType: v4.00+
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.601

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Kara,{font},{fontsize},&H{primary},&H{secondary},&H{outline},&H{back},{bold},{italic},0,0,100,100,0,0,1,{outline_w},{shadow_w},{align},{mL},{mR},{mV},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

def secs_to_ass(t: float) -> str:
    if t<0: t=0
    h=int(t//3600); m=int((t%3600)//60); s=int(t%60); cs=int(round((t-int(t))*100))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def chunk_for_ass(words: List[Dict], max_words: int)->List[List[Dict]]:
    return [words[i:i+max_words] for i in range(0, len(words), max_words)]

def make_kara_line(ch: List[Dict]):
    s=ch[0]["start"]; e=ch[-1]["end"]; parts=[]
    for w in ch:
        parts.append(rf"{{\k{duration_to_k_cs(w['end']-w['start'])}}}{w['word'].replace('{','(').replace('}',')')}")
    return s,e," ".join(parts)

def build_ass(aligned_words: List[Dict], style: dict, max_line_words: int)->str:
    header = ASS_HEADER.format(**style); lines=[header]
    for ch in chunk_for_ass(aligned_words, max_line_words):
        s,e,text = make_kara_line(ch)
        lines.append(f"Dialogue: 0,{secs_to_ass(s)},{secs_to_ass(e)},Kara,,0000,0000,0000,,{text}")
    return "\n".join(lines)

# ---------- main ----------
def main():
    ap=argparse.ArgumentParser(description="Forced-alignment karaoke captions (global-CTC first, chunked fallback)")
    ap.add_argument("--config", required=True)
    args=ap.parse_args()

    cfg=_abspath(args.config); _must_exist("Config", cfg)
    with open(cfg,"r",encoding="utf-8") as f: 
        C=yaml.safe_load(f)

    video=_abspath(C["paths"]["video"]); script=_abspath(C["paths"]["script"])
    _must_exist("Video", video); _must_exist("Script", script)

    base=os.path.splitext(os.path.basename(video))[0]
    out_ass=_abspath(C.get("output",{}).get("ass", f"{base}.ass"))
    out_video=_abspath(C.get("output",{}).get("video", f"{base}_karaoke.mp4"))

    # script
    drop_titles = bool(C.get("clean", {}).get("drop_title_lines", True))
    raw_text = load_and_clean_script(script, drop_title_lines=drop_titles)
    display_words = words_for_display(raw_text)
    if not display_words:
        print("No usable words in script after cleaning."); 
        sys.exit(2)

    # audio
    tmp=_abspath(C.get("advanced",{}).get("temp_audio", f"{base}_temp.wav"))
    sr=int(C.get("advanced",{}).get("sample_rate", 16000))
    wav, sr = extract_mono_wav(video, tmp, sr)

    # model
    device="cuda" if torch.cuda.is_available() else "cpu"
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device).eval()
    labels = bundle.get_labels()
    blank_id = labels.index("<blank>") if "<blank>" in labels else 0

    emissions, frame_dur = get_emissions(model, wav, sr, device)  # [T, C]
    T = emissions.shape[0]; audio_len = len(wav)/sr

    # config: alignment behavior
    A = C.get("align", {})
    mode = str(A.get("mode", "auto")).lower()   # "auto" | "global" | "chunked"
    min_char_hit_ratio = float(A.get("min_char_hit_ratio", 0.75))  # accept global if >= 75% chars timed
    highlight_linger_ms = int(C.get("refine",{}).get("highlight_linger_ms", 40))  # small linger after end

    # ---- compute a lenient VAD pass for sync targeting (find last spoken time) ----
    U = C.get("utterance",{})
    vad_for_sync = detect_pauses(
        wav, sr,
        frame_ms=int(U.get("frame_ms", 30)),
        hop_ms=int(U.get("hop_ms", 15)),
        # Be a bit more lenient than main VAD to catch quiet outro speech:
        energy_thresh_db=float(U.get("energy_thresh_db", -35.0)) - 5.0,
        min_pause_s=float(U.get("min_pause_s", 0.6)),
        max_utter_s=float(U.get("max_utter_s", 20.0))*2,
    )
    last_speech_end = vad_for_sync[-1][1] if vad_for_sync else audio_len

    # Build CTC text once
    ctc_full = normalize_for_labels(" ".join(display_words), labels)

    def postprocess_words(words: List[Dict]) -> List[Dict]:
        # global lead & clamp + monotone enforce + small linger
        S=C.get("sync",{})
        lead_ms=int(C.get("refine",{}).get("lead_ms", -100))
        max_shift_ms=int(C.get("refine",{}).get("max_shift_ms", 220))
        lead = max(-max_shift_ms, min(max_shift_ms, lead_ms))/1000.0
        linger = max(0, highlight_linger_ms)/1000.0

        out = [dict(w) for w in words]
        for w in out:
            w["start"] = max(0.0, w["start"] + lead)
            w["end"]   = min(audio_len, w["end"] + lead + linger)
            if w["end"] <= w["start"]: 
                w["end"] = w["start"] + 0.06

        # enforce monotonic non-overlapping
        for i in range(1, len(out)):
            if out[i]["start"] < out[i-1]["end"]:
                d = out[i-1]["end"] - out[i]["start"] + 1e-3
                out[i]["start"] += d; out[i]["end"] += d

        # --- improved linear warp to hit end target (speech_end or audio_end) ---
        if bool(S.get("linear_warp", True)) and len(out) >= 2:
            warp_target = str(S.get("warp_target", "speech_end")).lower()  # speech_end | audio_end
            end_margin  = float(S.get("end_margin_s", 0.35))
            max_warp_pct = float(S.get("max_warp_pct", 12.0))  # ±percent cap
            t0 = out[0]["start"]; tN = out[-1]["end"]

            if warp_target == "audio_end":
                target_last = max(0.0, audio_len - end_margin)
            else:  # speech_end (default)
                target_last = max(0.0, min(last_speech_end, audio_len) - end_margin)

            if tN > t0 + 1e-3 and abs(target_last - tN) > 1e-3:
                raw_scale = (target_last - t0) / (tN - t0)
                lo = 1.0 - (max_warp_pct/100.0)
                hi = 1.0 + (max_warp_pct/100.0)
                scale = float(np.clip(raw_scale, lo, hi))
                print(f"[warp] span {t0:.2f}->{tN:.2f}s -> target {target_last:.2f}s | raw_scale={raw_scale:.4f}, applied={scale:.4f}")
                for w in out:
                    w["start"] = t0 + (w["start"] - t0)*scale
                    w["end"]   = t0 + (w["end"]   - t0)*scale

        # final clamp
        for w in out:
            w["start"] = max(0.0, min(audio_len, w["start"]))
            w["end"]   = max(0.0, min(audio_len, w["end"]))
        return out

    # ---------- path 1: global alignment ----------
    def try_global():
        if not ctc_full:
            return None
        try:
            char_times = align_ctc_global(emissions, labels, ctc_full, blank_id, frame_dur)
        except Exception as e:
            print(f"[global] alignment failed: {e}")
            return None
        finite = int(np.isfinite(char_times).sum())
        hit_ratio = float(finite) / max(1, len(ctc_full))
        print(f"[global] char hit ratio: {hit_ratio:.3f} ({finite}/{len(ctc_full)})")
        if hit_ratio < min_char_hit_ratio:
            return None
        words = map_char_timings_to_words(ctc_full, char_times, display_words)
        return postprocess_words(words)

    # ---------- path 2: chunked alignment (fallback/opt-in) ----------
    def run_chunked():
        pauses = detect_pauses(
            wav, sr,
            frame_ms=int(U.get("frame_ms", 30)),
            hop_ms=int(U.get("hop_ms", 15)),
            energy_thresh_db=float(U.get("energy_thresh_db", -35.0)),
            min_pause_s=float(U.get("min_pause_s", 0.6)),
            max_utter_s=float(U.get("max_utter_s", 20.0)),
        )
        if not pauses: 
            pauses=[(0.0, audio_len)]
        word_chunks = words_to_chunks(display_words, pauses)

        pad_s=float(U.get("slice_pad_s", 0.6))
        max_chars=int(U.get("max_chars", 320))
        max_sub_sec=float(U.get("max_subchunk_s", 14.0))

        out=[]
        for (t0,t1), wchunk in zip(pauses, word_chunks):
            if not wchunk: 
                continue
            for sub_words, (s0,s1) in autosubchunk(wchunk, t0, t1, max_chars, max_sub_sec):
                if not sub_words: 
                    continue
                ss=max(0.0, s0 - pad_s); ee=min(audio_len, s1 + pad_s)
                fi0=int(round(ss/frame_dur)); fi1=int(round(ee/frame_dur))+1
                fi0=max(0, min(T-1, fi0)); fi1=max(fi0+1, min(T, fi1))
                em_slice = emissions[fi0:fi1]

                sub_ctc = normalize_for_labels(" ".join(sub_words), labels)
                if not sub_ctc:
                    # uniform fallback if nothing survives mapping
                    dur=max(0.1, s1-s0); step=dur/len(sub_words); t=s0
                    for w in sub_words:
                        out.append({"word": w, "start": t, "end": t+max(0.08, step*0.8)})
                        t += step
                    continue

                try:
                    char_times = align_ctc_slice(em_slice, labels, sub_ctc, blank_id, frame_dur, ss)
                    words = map_char_timings_to_words(sub_ctc, char_times, sub_words)
                    out.extend(words)
                except Exception as e:
                    # robust fallback: uniform pacing in (s0,s1)
                    dur=max(0.1, s1-s0); step=dur/len(sub_words); t=s0
                    for w in sub_words:
                        out.append({"word": w, "start": t, "end": t+max(0.08, step*0.8)})
                        t += step
                    continue

        if not out:
            return None
        return postprocess_words(out)

    # choose alignment path
    A_mode = mode
    if A_mode == "global":
        aligned_words = try_global() or run_chunked()
    elif A_mode == "chunked":
        aligned_words = run_chunked()
    else:  # "auto"
        aligned_words = try_global() or run_chunked()

    if not aligned_words:
        print("Alignment produced no words."); 
        sys.exit(2)

    # ---- apply global A/V offset so ASS times match the VIDEO timeline ----
    sync_cfg = C.get("sync", {})
    auto_off = float(probe_av_offset_seconds(video)) if bool(sync_cfg.get("auto_stream_offset", True)) else 0.0
    manual_ms = int(sync_cfg.get("global_offset_ms", 0))
    global_off = auto_off + (manual_ms / 1000.0)
    if abs(global_off) > 1e-6:
        print(f"[sync] applying global offset {global_off:+.3f}s (auto {auto_off:+.3f}s, manual {manual_ms} ms)")
        for w in aligned_words:
            w["start"] += global_off
            w["end"]   += global_off

    # write ASS
    style={
        "font": C.get("style",{}).get("font","Arial"),
        "fontsize": C.get("style",{}).get("fontsize",34),
        "primary": C.get("style",{}).get("primary","00FFFFFF"),
        "secondary": C.get("style",{}).get("secondary","00666666"),
        "outline": C.get("style",{}).get("outline","00000000"),
        "back": C.get("style",{}).get("back","32000000"),
        "bold": -1 if C.get("style",{}).get("bold",True) else 0,
        "italic": -1 if C.get("style",{}).get("italic",False) else 0,
        "outline_w": C.get("style",{}).get("outline_w",3),
        "shadow_w": C.get("style",{}).get("shadow_w",0),
        "align": C.get("style",{}).get("align",2),
        "mL": C.get("style",{}).get("marginL",50),
        "mR": C.get("style",{}).get("marginR",50),
        "mV": C.get("style",{}).get("marginV",60),
    }
    max_line_words=int(C.get("layout",{}).get("max_line_words",8))
    ass_text = build_ass(aligned_words, style, max_line_words)
    os.makedirs(os.path.dirname(out_ass) or ".", exist_ok=True)
    with open(out_ass,"w",encoding="utf-8") as f: 
        f.write(ass_text)
    print(f"[WRITE] ASS -> {out_ass}")

    if bool(C.get("output",{}).get("burn_in", False)):
        enc=C.get("encode",{})
        vcodec=enc.get("vcodec","libx264"); crf=str(enc.get("crf",16))
        preset=enc.get("preset","slow"); tune=enc.get("tune","")
        pix=enc.get("pix_fmt","yuv420p"); a_copy=enc.get("audio_copy",True)
        extra=enc.get("extra_flags","-movflags +faststart")
        tune_flag=f"-tune {tune}" if tune else ""; a_flag="-c:a copy" if a_copy else "-c:a aac -b:a 192k"
        cmd=f'ffmpeg -y -i "{video}" -vf "ass={out_ass}" -c:v {vcodec} -crf {crf} -preset {preset} {tune_flag} -pix_fmt {pix} {a_flag} {extra} "{out_video}"'
        run(cmd); print(f"[DONE] Burned-in -> {out_video}")
    else:
        print("[DONE] Soft subs only. To burn later:")
        print(f'  ffmpeg -y -i "{video}" -vf "ass={out_ass}" -c:v libx264 -crf 16 -preset slow -pix_fmt yuv420p -c:a copy -movflags +faststart "{out_video}"')

    # cleanup
    try: 
        os.remove(tmp)
    except OSError: 
        pass

if __name__ == "__main__":
    main()
