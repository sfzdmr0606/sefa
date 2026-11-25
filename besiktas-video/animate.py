"""Bring the still photo to life with a dense warp field.

Uses the face mesh + hair/face/body masks from detect.py. Each frame is a
backward map (cv2.remap) built from:
  * head motion per person (sway, beat nods) pivoting at the neck
  * breathing (subtle torso scale)
  * hair blowing in the wind
  * eye blinks (upper-lid skin pulled over the eye + lash line)
  * background parallax (sea/mountains drift against the people)
All fields are smooth, so the warp never tears or leaves holes.
"""
import math
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

BEAT = 0.5
DROP = 2.0
FINAL = 14.0

# face-mesh landmark ids
CHIN, FOREHEAD = 152, 10
EYES = [(33, 133, 159, 145), (362, 263, 386, 374)]  # corner, corner, upper lid, lower lid


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def soft(mask, sigma):
    return np.clip(gaussian_filter(mask.astype(np.float32), sigma) * 1.4, 0, 1)


class LivePhoto:
    def __init__(self, det_path, shape):
        d = np.load(det_path, allow_pickle=True)
        cat = d["cat"]
        cat = cat[..., 0] if cat.ndim == 3 else cat
        self.H, self.W = H, W = shape
        self.gy, self.gx = np.mgrid[0:H, 0:W].astype(np.float32)
        gx, gy = self.gx, self.gy
        faces = [np.asarray(f, dtype=np.float32) for f in d["faces"]]
        person = cat != 0

        centers = [f.mean(0) for f in faces]
        sizes = [float(np.ptp(f[:, 1])) for f in faces]
        dist = np.stack([np.hypot(gx - c[0], gy - c[1]) / s for c, s in zip(centers, sizes)])
        owner = np.argmin(dist, 0)

        self.people = []
        for i, f in enumerate(faces):
            cx, cy = centers[i]
            size = sizes[i]
            chin = f[CHIN]
            pm = person & (owner == i)
            head = pm & np.isin(cat, (1, 3, 5)) & (gy < chin[1] + 0.12 * size) & (np.abs(gx - cx) < 1.3 * size)
            neck = pm & (gy >= chin[1]) & (np.abs(gx - cx) < 0.7 * size)
            neck_w = np.where(neck, np.clip(1 - (gy - chin[1]) / (0.55 * size), 0, 1), 0)
            w_head = np.maximum(soft(head, 0.05 * size), gaussian_filter(neck_w.astype(np.float32), 0.05 * size))
            hair = pm & (cat == 1)
            # hair farther from the face centre moves more
            reach = np.clip((np.hypot(gx - cx, gy - cy) / size - 0.35) * 1.6, 0, 1.4)
            w_hair = soft(hair, 4) * reach
            eyes = []
            for a, b, up, lo in EYES:
                pa, pb, pu, pl = f[a], f[b], f[up], f[lo]
                eyes.append(
                    dict(
                        c=(pa + pb) / 2 * 0.5 + (pu + pl) / 2 * 0.5,
                        hw=float(np.linalg.norm(pa - pb)) / 2 * 1.2,
                        h=max(float(pl[1] - pu[1]), 0.035 * size),
                        top=float(pu[1]),
                    )
                )
            self.people.append(
                dict(
                    center=(cx, cy), size=size, pivot=(float(chin[0]), float(chin[1] + 0.18 * size)),
                    w_head=w_head.astype(np.float32), w_body=soft(pm, 8).astype(np.float32),
                    w_hair=w_hair.astype(np.float32), eyes=eyes, base_y=float(H),
                )
            )
        self.w_bg = (1 - soft(person, 14)).astype(np.float32)

    # ----------------------------------------------------------- motion script
    @staticmethod
    def blink_amount(t, times):
        a = 0.0
        for bt in times:
            x = (t - bt) / 0.075
            if -1 < x < 1.6:
                a = max(a, 1 - abs(x) if x < 0 else max(0, 1 - (x / 1.6)))
        return smoothstep(a * 1.25)

    def motion(self, i, t):
        """(angle_deg, tx, ty, breath, blink) for person i at time t."""
        size = self.people[i]["size"]
        ph = [0.0, 1.9, 3.7][i % 3]
        ang = 1.6 * math.sin(2 * math.pi * t / 3.3 + ph) + 0.6 * math.sin(2 * math.pi * t / 1.7 + ph * 2)
        tx = 0.012 * size * math.sin(2 * math.pi * t / 4.1 + ph)
        ty = 0.0
        if DROP <= t < FINAL + 0.2:
            # head-bob on every beat, with a side-to-side tilt over two beats
            k = (t - DROP) / BEAT
            bob = math.sin(math.pi * (k % 1.0)) ** 2
            energy = 0.7 if t < 10 else 1.0
            ty += 0.028 * size * bob * energy
            ang += 2.2 * energy * math.sin(math.pi * k / 2 + ph)
        if t >= FINAL:
            # cheer: heads lift and tilt back at the final hit
            lift = math.exp(-((t - FINAL - 0.35) ** 2) / 0.08)
            ty -= 0.05 * size * lift
            ang += 3.0 * lift * (1 if i % 2 else -1)
        breath = 0.009 * math.sin(2 * math.pi * t / 3.6 + ph)
        blinks = [[2.7, 5.1, 9.3, 11.9, 15.2], [3.4, 7.0, 10.6, 13.1], []][i % 3]
        return ang, tx, ty, breath, self.blink_amount(t, blinks)

    # ----------------------------------------------------------- warp
    def maps(self, t):
        gx, gy = self.gx, self.gy
        dx = np.zeros_like(gx)
        dy = np.zeros_like(gy)
        # background parallax: scenery drifts slowly against the people
        bgx = 10 * math.sin(2 * math.pi * t / 7.5) + 4 * math.sin(2 * math.pi * t / 2.9)
        bgy = 3 * math.sin(2 * math.pi * t / 5.3)
        dx += self.w_bg * bgx
        dy += self.w_bg * bgy
        params = [self.motion(i, t) for i in range(len(self.people))]
        for p, (ang, tx, ty, breath, _) in zip(self.people, params):
            px, py = p["pivot"]
            a = math.radians(ang)
            ca, sa = math.cos(a), math.sin(a)
            rx = ca * (gx - px) + sa * (gy - py) + px
            ry = -sa * (gx - px) + ca * (gy - py) + py
            w = p["w_head"]
            dx += w * (rx - gx - tx)
            dy += w * (ry - gy - ty)
            # breathing: torso scales up from the bottom edge
            sy = p["base_y"] + (gy - p["base_y"]) / (1 + breath)
            dy += p["w_body"] * (1 - w) * (sy - gy)
        qx, qy = gx + dx, gy + dy

        # local effects evaluated in the un-moved photo's coordinates (q)
        ex = np.zeros_like(gx)
        ey = np.zeros_like(gy)
        lids = []
        for p, (_, _, _, _, blink) in zip(self.people, params):
            hair = cv2.remap(p["w_hair"], qx, qy, cv2.INTER_LINEAR)
            ph = p["center"][0] * 0.01
            wave = np.sin(qy * 0.045 - t * 7.0 + ph) + 0.5 * np.sin(qx * 0.03 + t * 4.3)
            amp = 0.012 * p["size"] * (1 + 0.6 * math.sin(t * 1.3 + ph))
            ex += hair * amp * wave
            ey += hair * amp * 0.35 * np.cos(qx * 0.05 - t * 6.0)
            if blink > 0.01:
                for e in p["eyes"]:
                    cx, cy = e["c"]
                    u = (qx - cx) / e["hw"]
                    v = (qy - cy) / (e["h"] * 1.15)
                    m = smoothstep((1 - (u * u + v * v)) * 2.5) * blink
                    # closed lid: skin tone, a touch darker toward the lash line
                    grad = 1 - 0.12 * np.clip((qy - (cy - e["h"])) / (2 * e["h"]), 0, 1)
                    lash = np.exp(-(((qy - (cy + 0.3 * e["h"])) / max(1.3, 0.16 * e["h"])) ** 2))
                    lash *= smoothstep((1 - u * u) * 2.5) * blink
                    lids.append((m, grad * (1 - 0.55 * lash), (cx, cy + 1.6 * e["h"]), e["h"]))
        return (qx + ex).astype(np.float32), (qy + ey).astype(np.float32), lids

    def warp(self, img, maps):
        mx, my, lids = maps
        out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if lids:
            out = out.astype(np.float32)
            for m, shade, (sx, sy), h in lids:
                r = max(2, int(h * 0.25))
                patch = img[int(sy) - r : int(sy) + r + 1, int(sx) - 3 * r : int(sx) + 3 * r + 1]
                col = np.median(patch.reshape(-1, 3).astype(np.float32), 0) * 0.93
                lid = col[None, None, :] * shade[..., None]
                out = out * (1 - m[..., None]) + lid * m[..., None]
            out = out.clip(0, 255).astype(np.uint8)
        return out
