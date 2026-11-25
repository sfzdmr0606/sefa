"""Original stadium-style fan track (120 BPM, 16 s) for the Beşiktaş family short.

Everything is synthesized here: drums, claps, brass melody, crowd noise and a
"Beşiktaş!" crowd chant built from layered Turkish espeak-ng voices.
No copyrighted recording is used.
"""
import subprocess, wave, sys, os
import numpy as np

SR = 44100
BPM = 120
BEAT = 60 / BPM
DUR = 16.0
N = int(SR * DUR)
rng = np.random.default_rng(1903)
OUT = sys.argv[1] if len(sys.argv) > 1 else "build/music.wav"
TMP = os.path.join(os.path.dirname(OUT) or ".", "voices")
os.makedirs(TMP, exist_ok=True)

L = np.zeros(N)
R = np.zeros(N)


def place(sig, t, gain=1.0, pan=0.0):
    i = int(t * SR)
    if i >= N:
        return
    sig = sig[: N - i] * gain
    L[i : i + len(sig)] += sig * np.sqrt(0.5 * (1 - pan))
    R[i : i + len(sig)] += sig * np.sqrt(0.5 * (1 + pan))


def fft_filter(x, lo=None, hi=None):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    m = np.ones_like(f)
    if lo:
        m *= 1 / (1 + (lo / np.maximum(f, 1e-3)) ** 4)
    if hi:
        m *= 1 / (1 + (f / hi) ** 4)
    return np.fft.irfft(X * m, len(x))


def reverb(x, secs=1.2, mix=0.3):
    n = int(secs * SR)
    ir = rng.standard_normal(n) * np.exp(-np.linspace(0, 7, n))
    ir = fft_filter(ir, hi=5000)
    ir /= np.sqrt(np.sum(ir**2))
    size = len(x) + n
    y = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)[: len(x)]
    return x * (1 - mix) + y * mix


# ---------------------------------------------------------------- instruments
def kick():
    t = np.arange(int(0.35 * SR)) / SR
    f = 50 + 110 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)


def davul():  # big Turkish bass drum, boomy
    t = np.arange(int(0.6 * SR)) / SR
    f = 65 + 60 * np.exp(-t * 18)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 5)
    slap = fft_filter(rng.standard_normal(len(t)), lo=300, hi=3000) * np.exp(-t * 60) * 0.4
    return body + slap


def clap():
    t = np.arange(int(0.25 * SR)) / SR
    env = np.zeros_like(t)
    for d in (0, 0.011, 0.022, 0.031):  # many hands, slightly apart
        env += np.where(t >= d, np.exp(-(t - d) * 45), 0)
    return fft_filter(rng.standard_normal(len(t)), lo=800, hi=6000) * env * 0.6


def crowd_clap():  # a stadium of claps = many smeared claps
    out = np.zeros(int(0.35 * SR))
    for _ in range(25):
        c = clap()
        o = int(abs(rng.normal(0, 0.018)) * SR)
        out[o : o + len(c)] += c[: len(out) - o] * rng.uniform(0.3, 1)
    return out / 8


def snare_hit():
    t = np.arange(int(0.12 * SR)) / SR
    return fft_filter(rng.standard_normal(len(t)), lo=1500, hi=9000) * np.exp(-t * 40)


def boom():  # cinematic impact
    t = np.arange(int(2.5 * SR)) / SR
    f = 35 + 90 * np.exp(-t * 8)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.2)
    noise = fft_filter(rng.standard_normal(len(t)), hi=2500) * np.exp(-t * 5) * 0.5
    return sub + noise


def note_freq(name):
    names = {"C": -9, "D": -7, "E": -5, "F": -4, "G": -2, "A": 0, "B": 2}
    semis = names[name[0]]
    rest = name[1:]
    if rest.startswith("b"):
        semis -= 1
        rest = rest[1:]
    octave = int(rest)
    return 440 * 2 ** ((semis + (octave - 4) * 12) / 12)


def brass(freq, dur):
    n = int((dur + 0.15) * SR)
    t = np.arange(n) / SR
    vib = 1 + 0.004 * np.sin(2 * np.pi * 5.5 * t) * np.clip(t * 3, 0, 1)
    ph = 2 * np.pi * np.cumsum(freq * vib) / SR
    bright = 0.35 + 0.65 * np.exp(-t * 6)  # brassy attack then mellow
    sig = sum((bright ** (k / 4)) * np.sin(k * ph) / k for k in range(1, 14))
    env = np.clip(t / 0.03, 0, 1) * np.clip((dur + 0.15 - t) / 0.15, 0, 1)
    return sig * env * 0.5


def bass(freq, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    ph = 2 * np.pi * freq * t
    sig = np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.15 * np.sin(3 * ph)
    return sig * np.exp(-t * 2.5) * np.clip(t / 0.01, 0, 1) * np.clip((dur - t) / 0.03, 0, 1)


# ---------------------------------------------------------------- chant voices
def read_wav(path):
    with wave.open(path) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(float) / 32768
    # resample to SR
    tt = np.arange(int(len(x) * SR / sr)) / SR
    x = np.interp(tt, np.arange(len(x)) / sr, x)
    nz = np.where(np.abs(x) > 0.02)[0]
    return x[nz[0] : nz[-1] + 1] if len(nz) else x


def chant_sample():
    layers = []
    k = 0
    for pitch in (15, 30, 40, 50, 60, 75):
        for speed in (165, 185):
            p = os.path.join(TMP, f"v{k}.wav")
            subprocess.run(
                ["espeak-ng", "-v", "tr", "-s", str(speed), "-p", str(pitch), "-a", "180", "-w", p, "Beşiktaş"],
                check=True,
            )
            layers.append(read_wav(p))
            k += 1
    length = int(1.1 * SR)
    out = np.zeros(length)
    for x in layers:
        for _ in range(2):  # pitch/formant shift by resampling -> more people
            r = rng.uniform(0.85, 1.12)
            y = np.interp(np.arange(0, len(x), r), np.arange(len(x)), x)
            o = int(rng.uniform(0, 0.045) * SR)
            y = y[: length - o]
            out[o : o + len(y)] += y * rng.uniform(0.5, 1.0)
    out = fft_filter(out, lo=120, hi=4500)
    out = np.tanh(out / np.max(np.abs(out)) * 2.5)  # shouty saturation
    return reverb(out, 1.4, 0.45)


# ---------------------------------------------------------------- crowd bed
crowd = fft_filter(rng.standard_normal(N), lo=250, hi=2200)
# slow murmur modulation
mod = np.interp(np.arange(N), np.linspace(0, N, 60), rng.uniform(0.6, 1.0, 60))
crowd *= mod
t_all = np.arange(N) / SR
crowd_env = np.clip(t_all / 2.0, 0, 1) * 0.10 + 0.06
crowd_env += np.where(t_all > 14, 0.22 * np.exp(-(t_all - 14) * 0.6), 0)  # final cheer
crowd *= crowd_env / np.std(crowd)
L += crowd * 0.6
R += np.roll(crowd, 900) * 0.6

# ---------------------------------------------------------------- arrangement
DROP = 2.0           # beat 4
END = 14.0           # beat 28
b = lambda n: n * BEAT

# intro: snare roll riser + noise sweep
for i in range(16):
    tt = i * 0.125
    place(snare_hit(), tt, 0.15 + 0.5 * i / 16, pan=rng.uniform(-0.3, 0.3))
riser_n = int(DROP * SR)
riser = fft_filter(rng.standard_normal(riser_n), lo=500, hi=7000) * np.linspace(0, 1, riser_n) ** 2 * 0.15
place(riser, 0, 1)

K, D, C = kick(), davul(), crowd_clap()
for beat in range(4, 28):
    t = b(beat)
    place(K, t, 0.9)
    if beat % 2 == 0:
        place(D, t, 0.55, -0.2)
    else:
        place(D, t + BEAT * 0.5, 0.35, 0.2)  # syncopated davul
    place(snare_hit(), t + BEAT * 0.5, 0.12, 0.4)  # off-beat hats feel

# chant + clap phrase every 8 beats: "BE-ŞİK-TAŞ!  clap clap | BE-ŞİK-TAŞ! clap clap clap"
CH = chant_sample()
for start in range(4, 28, 8):
    place(CH, b(start) - 0.02, 0.85)
    for cb in (2, 3):
        place(C, b(start + cb), 1.0, rng.uniform(-0.3, 0.3))
    place(CH, b(start + 4) - 0.02, 0.85)
    for cb in (6, 6.5, 7):
        place(C, b(start + cb), 1.0, rng.uniform(-0.3, 0.3))

# bass line (D minor: D - Bb - C - A), one chord per 2 beats... per bar
roots = ["D2", "D2", "Bb1", "C2", "D2", "D2", "Bb1", "A1"]
for i, beat in enumerate(range(4, 28, 4)):
    r = note_freq(roots[i % len(roots)])
    for k in range(4):
        place(bass(r, BEAT * 0.9), b(beat + k), 0.35)

# original brass fanfare melody (enters after the first chant phrase)
melody = [
    ("D5", 1), ("D5", 0.5), ("F5", 0.5), ("A5", 1.5), ("G5", 0.5),
    ("F5", 1), ("E5", 0.5), ("F5", 0.5), ("D5", 2),
    ("C5", 1), ("D5", 0.5), ("E5", 0.5), ("F5", 1.5), ("E5", 0.5),
    ("D5", 1), ("C5", 0.5), ("A4", 0.5), ("D5", 2),
]
for section in (12,):
    t = b(section)
    for name, dur in melody:
        f = note_freq(name)
        hit = brass(f, dur * BEAT * 0.92) + 0.6 * brass(f / 2, dur * BEAT * 0.92)
        place(hit, t, 0.32, 0.15)
        t += dur * BEAT

# drop + final impacts
place(boom(), DROP, 0.9)
place(brass(note_freq("D4"), 1.2) + brass(note_freq("A4"), 1.2) + brass(note_freq("D5"), 1.2), DROP, 0.35)
place(boom(), END, 1.0)
place(CH, END + 0.05, 1.1)
place(CH, END + 0.9, 0.7)
chord = sum(brass(note_freq(n), 1.8) for n in ("D4", "F4", "A4", "D5"))
place(chord, END, 0.3)

# ---------------------------------------------------------------- master
L, R = reverb(L, 0.8, 0.12), reverb(R, 0.8, 0.12)
st = np.stack([L, R], 1)
st /= np.max(np.abs(st)) + 1e-9
st = np.tanh(st * 1.6) / np.tanh(1.6)
fade = np.ones(N)
fn = int(0.6 * SR)
fade[-fn:] = np.linspace(1, 0, fn)
st *= fade[:, None] * 0.93
pcm = (st * 32767).astype(np.int16)
with wave.open(OUT, "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print("wrote", OUT)
