"""Original stadium anthem (120 BPM, 16 s) - v2.

All sound is synthesized (no recordings, no copyrighted melody):
  * crowd chant "BE-ŞİK-TAŞ!" from ~50 formant-synthesized shouting voices
  * crowd singing an original "oh-oh-oh" melody
  * Turkish davul (deep boom + stick), kick, stadium claps, tom fills
  * brass section (trumpets + trombones), bass
  * crowd roar, whistles, long stadium reverb
Usage: python3 make_music.py out.wav
"""
import sys, wave
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve

SR = 44100
BPM = 120
BEAT = 60 / BPM
DUR = 16.0
N = int(SR * DUR)
rng = np.random.default_rng(7)
OUT = sys.argv[1] if len(sys.argv) > 1 else "build/music.wav"

L = np.zeros(N)
R = np.zeros(N)
bus_rev = np.zeros((N, 2))  # send to stadium reverb


def b(n):
    return n * BEAT


def place(sig, t, gain=1.0, pan=0.0, rev=0.0):
    i = int(t * SR)
    if i >= N or i < 0:
        return
    sig = sig[: N - i] * gain
    gl, gr = np.sqrt(0.5 * (1 - pan)), np.sqrt(0.5 * (1 + pan))
    L[i : i + len(sig)] += sig * gl
    R[i : i + len(sig)] += sig * gr
    if rev:
        bus_rev[i : i + len(sig), 0] += sig * gl * rev
        bus_rev[i : i + len(sig), 1] += sig * gr * rev


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], btype="band", fs=SR, output="sos"), x)


def lp(x, f, order=2):
    return sosfilt(butter(order, f, btype="low", fs=SR, output="sos"), x)


def hp(x, f, order=2):
    return sosfilt(butter(order, f, btype="high", fs=SR, output="sos"), x)


def env_adsr(n, a=0.01, d=0.1, s=0.7, r=0.1):
    t = np.arange(n) / SR
    dur = n / SR
    e = np.where(t < a, t / a, s + (1 - s) * np.exp(-(t - a) / max(d, 1e-3)))
    e *= np.clip((dur - t) / r, 0, 1)
    return e


# ------------------------------------------------------------------ voices
VOWELS = {  # shouted formants (Hz) and relative gains
    "a": ((850, 1300, 2600), (1.0, 0.7, 0.25)),
    "e": ((600, 1850, 2600), (1.0, 0.55, 0.3)),
    "i": ((380, 2200, 3000), (1.0, 0.45, 0.3)),
    "o": ((560, 950, 2500), (1.0, 0.6, 0.15)),
}


def formant(x, vowel, shift=1.0):
    fs, gs = VOWELS[vowel]
    y = np.zeros_like(x)
    for f, g, bw in zip(fs, gs, (130, 170, 230)):
        f *= shift
        y += g * bp(x, f - bw / 2, f + bw / 2)
    return y


def voice_source(n, f0, vib=0.0, jitter=0.01, breath=0.25):
    t = np.arange(n) / SR
    drift = np.interp(t, np.linspace(0, n / SR, 8), 1 + rng.normal(0, jitter, 8))
    f = f0 * drift * (1 + vib * np.sin(2 * np.pi * rng.uniform(4.5, 6) * t + rng.uniform(0, 6)))
    ph = np.cumsum(f) / SR
    saw = 2 * (ph % 1) - 1
    return saw + breath * rng.standard_normal(n)


def crowd(segments, total, n_voices=48, base=(165, 300), female=0.3, shout=True):
    """segments: list of (t0, dur, kind, pitch_ratio) where kind is a vowel or
    consonant: 'sh', 'k', 't', 'b'. Returns stereo (n, 2) crowd recording."""
    n = int(total * SR)
    out = np.zeros((n, 2))
    vowel_names = sorted({k for _, _, k, _ in segments if k in VOWELS})
    for v in range(n_voices):
        is_f = rng.random() < female
        f0 = (base[1] if is_f else base[0]) * 2 ** rng.normal(0, 0.09)
        shift = 1.15 if is_f else 1.0
        lag = abs(rng.normal(0, 0.022))
        amp = rng.uniform(0.4, 1.0)
        pan = rng.uniform(-0.9, 0.9)
        src = voice_source(n, f0, vib=0.004 if not shout else 0.0, jitter=0.012, breath=0.35 if shout else 0.12)
        # pitch contour per segment
        pitch = np.ones(n)
        sig = np.zeros(n)
        for vn in vowel_names:
            wv = np.zeros(n)
            for t0, dur, k, pr in segments:
                if k != vn:
                    continue
                i0 = int((t0 + lag) * SR)
                m = int(dur * SR)
                if i0 >= n:
                    continue
                m = min(m, n - i0)
                wv[i0 : i0 + m] = env_adsr(m, 0.025, 0.12, 0.8 if shout else 0.95, 0.06)
                pitch[i0 : i0 + m] = pr
            if wv.any():
                # re-pitch the source by resampling the phase (cheap: regenerate)
                s = voice_source(n, f0, vib=0.0 if shout else 0.006, jitter=0.012, breath=0.35 if shout else 0.12)
                if not np.allclose(pitch, 1):
                    t = np.arange(n) / SR
                    ph = np.cumsum(f0 * pitch * (1 + rng.normal(0, 0.003))) / SR
                    s = 2 * (ph % 1) - 1 + (0.35 if shout else 0.12) * rng.standard_normal(n)
                sig += formant(s, vn, shift) * wv
        for t0, dur, k, _ in segments:  # consonants
            i0 = int((t0 + lag) * SR)
            m = int(dur * SR)
            if i0 >= n or k in VOWELS:
                continue
            m = min(m, n - i0)
            noise = rng.standard_normal(m)
            tt = np.arange(m) / SR
            if k == "sh":
                c = bp(noise, 1800 * shift, 6000) * env_adsr(m, 0.02, 0.05, 0.8, 0.03) * 0.28
            elif k in ("k", "t"):
                c = hp(noise, 1500 if k == "k" else 2500) * np.exp(-tt * 90) * 0.4
            else:  # b
                c = lp(noise, 400) * np.exp(-tt * 60) * 0.5
            sig[i0 : i0 + m] += c
        sig *= amp
        out[:, 0] += sig * np.sqrt(0.5 * (1 - pan))
        out[:, 1] += sig * np.sqrt(0.5 * (1 + pan))
    out = hp(out.T, 110).T
    return out / (np.max(np.abs(out)) + 1e-9)


def chant_besiktas(scale=1.0):
    """BE - ŞİK - TAŞ! on an eighth, eighth, long note."""
    e = BEAT / 2
    seg = [
        (0.00, 0.03, "b", 1.0), (0.02, e * 0.85, "e", 1.0),
        (e - 0.04, 0.09, "sh", 1.0), (e + 0.04, e * 0.55, "i", 1.0), (2 * e - 0.07, 0.03, "k", 1.0),
        (2 * e, 0.025, "t", 1.0), (2 * e + 0.02, BEAT * 1.1 * scale, "a", 0.9), (2 * e + BEAT * 1.1 * scale, 0.16, "sh", 1.0),
    ]
    return crowd(seg, 2 * e + BEAT * 1.1 * scale + 0.4, n_voices=56, base=(190, 330), female=0.3, shout=True)


def crowd_song(notes, n_voices=44):
    """notes: (start_beat, beats, midi, vowel). Sung 'oh-oh' melody, men an octave below women."""
    total = max(s + d for s, d, _, _ in notes) * BEAT + 0.5
    base_midi = notes[0][2]
    segs = [(s * BEAT, d * BEAT * 0.95, v, 2 ** ((m - base_midi) / 12)) for s, d, m, v in notes]
    f_base = 440 * 2 ** ((base_midi - 69) / 12)
    return crowd(segs, total, n_voices=n_voices, base=(f_base / 2, f_base), female=0.35, shout=False)


# ------------------------------------------------------------------ drums
def davul_boom():
    n = int(0.9 * SR)
    t = np.arange(n) / SR
    f = 55 + 70 * np.exp(-t * 22)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 4.0)
    skin = bp(rng.standard_normal(n), 120, 900) * np.exp(-t * 25) * 0.5
    beater = hp(rng.standard_normal(n), 1500) * np.exp(-t * 200) * 0.25
    return body + skin + beater


def davul_tak():  # thin stick on the other head
    n = int(0.18 * SR)
    t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * 260 * t) * np.exp(-t * 35) * 0.4
    return tone + bp(rng.standard_normal(n), 1800, 7000) * np.exp(-t * 55)


def kick():
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    f = 48 + 120 * np.exp(-t * 35)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 8) + hp(rng.standard_normal(n), 3000) * np.exp(-t * 300) * 0.2


def tom(freq):
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    f = freq * (1 + 0.6 * np.exp(-t * 30))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7) + bp(rng.standard_normal(n), 200, 2500) * np.exp(-t * 40) * 0.3


def stadium_claps():
    n = int(0.4 * SR)
    out = np.zeros(n)
    for _ in range(60):
        m = int(0.06 * SR)
        tt = np.arange(m) / SR
        c = bp(rng.standard_normal(m), rng.uniform(700, 1300), rng.uniform(3000, 6000)) * np.exp(-tt * rng.uniform(60, 120))
        o = int(abs(rng.normal(0, 0.016)) * SR)
        out[o : o + m] += c * rng.uniform(0.3, 1)
    return out / np.max(np.abs(out))


def boom():
    n = int(3.0 * SR)
    t = np.arange(n) / SR
    f = 30 + 80 * np.exp(-t * 6)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 1.6)
    crash = hp(rng.standard_normal(n), 3000) * np.exp(-t * 2.2) * 0.35
    return sub + crash


# ------------------------------------------------------------------ tonal
def mf(m):
    return 440 * 2 ** ((m - 69) / 12)


def brass_note(m, dur, bright=1.0):
    n = int((dur + 0.12) * SR)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for det in (-6, 0, 7):  # cents
        f = mf(m) * 2 ** (det / 1200) * (1 + 0.003 * np.sin(2 * np.pi * 5.3 * t) * np.clip(t * 4 - 0.6, 0, 1))
        ph = np.cumsum(f) / SR
        sig += 2 * (ph % 1) - 1
    e = env_adsr(n, 0.035, 0.18, 0.75, 0.1)
    # brassy filter sweep: lowpass opening on the attack, done in two bands
    bright_part = hp(sig, 1500) * (np.exp(-t * 5) * 0.8 + 0.25) * bright
    body = lp(sig, 1600)
    return (body + bright_part) * e * 0.25


def bass_note(m, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    ph = 2 * np.pi * mf(m) * t
    return (np.sin(ph) + 0.4 * np.sin(2 * ph)) * env_adsr(n, 0.005, 0.2, 0.6, 0.04)


# ------------------------------------------------------------------ crowd bed
t_all = np.arange(N) / SR
roar = bp(rng.standard_normal(N), 250, 2800, 2)
roar2 = bp(rng.standard_normal(N), 250, 2800, 2)
mod = np.interp(t_all, np.linspace(0, DUR, 50), rng.uniform(0.75, 1.0, 50))
lvl = 0.05 + 0.08 * np.clip(t_all / 2.0, 0, 1)
lvl += np.where(t_all > 14, 0.25 * np.exp(-(t_all - 14) * 0.7), 0)
L += roar * mod * lvl / np.std(roar)
R += roar2 * mod * lvl / np.std(roar2)
for wt in (0.4, 1.1, 14.4, 15.0):  # whistles
    n = int(0.45 * SR)
    tt = np.arange(n) / SR
    f = 2300 + 500 * np.sin(np.pi * tt / 0.45)
    place(np.sin(2 * np.pi * np.cumsum(f) / SR) * env_adsr(n, 0.03, 0.2, 0.8, 0.1), wt, 0.05, rng.uniform(-0.8, 0.8), 0.5)

# ------------------------------------------------------------------ arrangement
DROP, FINAL = 2.0, 14.0

# intro: davul roll crescendo + rising crowd "ooo"
for i in range(16):
    tt = i * 0.125
    place(davul_tak(), tt, 0.1 + 0.5 * (i / 16) ** 1.5, 0.3)
    if i % 2 == 0:
        place(tom(90 + i * 4), tt, 0.15 + 0.4 * i / 16, -0.2)
ooo = crowd_song([(0, 3.6, 57, "o")], n_voices=40)
n = len(ooo)
rise = np.linspace(0, 1, n) ** 1.5
place(ooo[:, 0] * rise, 0.1, 0.35, -0.5, 0.6)
place(ooo[:, 1] * rise, 0.1, 0.35, 0.5, 0.6)

# groove
BOOM_D, TAK, K, CL = davul_boom(), davul_tak(), kick(), stadium_claps()
for beat in range(4, 28):
    t = b(beat)
    place(K, t, 0.55)
    if beat % 2 == 0:
        place(BOOM_D, t, 0.7, -0.1, 0.2)
    else:
        place(BOOM_D, t, 0.45, -0.1, 0.2)
    place(TAK, t + BEAT / 2, 0.35, 0.35, 0.2)
    if beat % 4 == 3:
        place(TAK, t + BEAT * 0.75, 0.25, 0.35, 0.2)

# chant: BE-ŞİK-TAŞ!  (clap clap clap) every bar, beats 4..13
CH = chant_besiktas()
CH_BIG = chant_besiktas(1.6)
for bar_beat in (4, 8, 12):
    place(CH[:, 0], b(bar_beat), 0.95, -0.4, 0.5)
    place(CH[:, 1], b(bar_beat), 0.95, 0.4, 0.5)
    for cb in (2, 2.5, 3):
        place(CL, b(bar_beat + cb), 0.65, rng.uniform(-0.3, 0.3), 0.35)

# bass + brass from beat 12
prog = [50, 50, 46, 48, 50, 50, 46, 45, 50, 50, 46, 48]  # D D Bb C ...
for i, beat in enumerate(range(4, 28, 2)):
    for k in range(4):
        place(bass_note(prog[i] - 12, BEAT * 0.45), b(beat) + k * BEAT / 2, 0.35 if beat >= 8 else 0.2)

melody = [  # (beat offset, beats, midi) original fanfare in D minor
    (0, 1, 62), (1, 0.5, 62), (1.5, 0.5, 65), (2, 1, 69), (3, 1, 67),
    (4, 1, 65), (5, 0.5, 64), (5.5, 0.5, 65), (6, 2, 62),
    (8, 1, 60), (9, 0.5, 62), (9.5, 0.5, 64), (10, 1, 65), (11, 1, 64),
    (12, 1, 62), (13, 0.5, 60), (13.5, 0.5, 57), (14, 2, 62),
]
start = 12
for off, d, m in melody:
    place(brass_note(m + 12, d * BEAT * 0.9), b(start + off), 0.28, 0.25, 0.35)
    place(brass_note(m, d * BEAT * 0.9, 0.6), b(start + off), 0.28, -0.25, 0.35)

# crowd sings along "oh - oh - oh" with the second half of the melody
song = crowd_song([(off, d, m, "o" if i % 4 != 3 else "a") for i, (off, d, m) in enumerate(melody[9:])], n_voices=44)
place(song[:, 0], b(start + 8), 0.75, -0.5, 0.6)
place(song[:, 1], b(start + 8), 0.75, 0.5, 0.6)
# and the chant answers in the gap
place(CH[:, 0], b(20), 0.8, -0.4, 0.5)
place(CH[:, 1], b(20), 0.8, 0.4, 0.5)

# tom fill into the final hit
for i in range(8):
    place(tom(150 - i * 11), b(27) + i * BEAT / 8 - BEAT / 2, 0.45 + i * 0.04, -0.4 + i * 0.1, 0.2)

# finale
place(boom(), FINAL, 0.9, 0, 0.3)
place(BOOM_D, FINAL, 0.9, 0, 0.3)
place(CH_BIG[:, 0], FINAL + 0.05, 1.2, -0.5, 0.6)
place(CH_BIG[:, 1], FINAL + 0.05, 1.2, 0.5, 0.6)
for m in (50, 57, 62, 65, 69):
    place(brass_note(m, 1.6), FINAL, 0.2, rng.uniform(-0.4, 0.4), 0.4)

# ------------------------------------------------------------------ master
n_ir = int(2.4 * SR)
ir_t = np.arange(n_ir) / SR
irs = []
for _ in range(2):
    ir = rng.standard_normal(n_ir) * np.exp(-ir_t * 2.9)
    ir[: int(0.03 * SR)] = 0  # pre-delay
    ir = lp(ir, 5000)
    irs.append(ir / np.sqrt(np.sum(ir**2)))
wetL = fftconvolve(bus_rev[:, 0], irs[0])[:N]
wetR = fftconvolve(bus_rev[:, 1], irs[1])[:N]
L += wetL * 0.55
R += wetR * 0.55
st = np.stack([L, R], 1)
st = hp(st.T, 30).T
st /= np.max(np.abs(st)) + 1e-9
st = np.tanh(st * 2.0) / np.tanh(2.0)  # glue / loudness
fade = np.ones(N)
fn = int(0.8 * SR)
fade[-fn:] = np.linspace(1, 0, fn) ** 1.5
st *= fade[:, None] * 0.94
with wave.open(OUT, "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((st * 32767).astype(np.int16).tobytes())
print("wrote", OUT)
