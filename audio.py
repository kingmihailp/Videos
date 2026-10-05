"""Voice-over (espeak-ng), procedural score + ambience, and ASS subtitles."""
import os
import subprocess
import wave

import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve, resample_poly

SR = 44100
DUR = 180

# (start second, text) - copy is built from the brief; replace/extend freely.
VO = [
    (2.5, "Вечер. Тишина кабинета. Рамки с бабочками, коробки с коллекциями, сачки у стены. Здесь рождаются большие открытия."),
    (14.5, "Flora0world HUB. Дом для энтомологов, природоведов и киперов."),
    (22.5, "Здесь делятся находками, учатся определять виды и говорят о жизни своих питомцев."),
    (31.0, "Здесь можно задать вопрос, показать свою коллекцию и найти тех, кто горит тем же."),
    (39.0, "Спокойно, без спешки, с любовью к живому."),
    (47.5, "Добро пожаловать!"),
    (57.5, "Глава первая. Ловля. Лёгкий сачок, один точный взмах, и редкая гостья на мгновение в твоих руках. Мы наблюдаем, фотографируем и отпускаем."),
    (73.0, "Лучшие находки не случайны. Нужно знать время, погоду и растения, на которые прилетают насекомые."),
    (87.0, "Глава вторая. Ночь. Белое полотно, яркий свет, и ночные бабочки сами приходят на огонёк."),
    (97.0, "Здесь делятся способами ловли, схемами светоловушек и снимками самых необычных гостей."),
    (112.0, "Глава третья. Карта. Источники, места находок и маршруты. Каждая точка на карте это чья-то история."),
    (124.5, "Леса, горы, побережья и тропики. Отмечай свои места и находи новые."),
    (136.0, "Опыт сообщества становится маршрутом, а маршрут приключением."),
    (147.0, "Глава четвёртая. Путешествия. Дорога ведёт туда, где живёт редкое, и рядом всегда найдутся единомышленники."),
    (158.5, "Вместе дальше. Вместе интереснее."),
    (167.0, "Каждое насекомое это целая вселенная. Присмотрись."),
    (174.9, "Flora0world HUB. Лови. Наблюдай. Делись."),
]


def sos(kind, fc, order=2):
    return butter(order, fc, kind, fs=SR, output="sos")


def lp(x, fc):
    return sosfilt(sos("low", fc), x)


def hp(x, fc):
    return sosfilt(sos("high", fc), x)


def tts(text, speed, path):
    subprocess.run(["espeak-ng", "-v", "ru+m3", "-s", str(speed), "-p", "38", "-a", "170", "-g", "2", "-w", path, text], check=True)
    with wave.open(path) as w:
        n, sr = w.getnframes(), w.getframerate()
        x = np.frombuffer(w.readframes(n), np.int16).astype(np.float32) / 32768
    return x, sr


def make_voice(out):
    voice = np.zeros(DUR * SR, np.float32)
    timing = []
    for i, (t0, text) in enumerate(VO):
        limit = (VO[i + 1][0] if i + 1 < len(VO) else DUR - 0.2) - 0.4 - t0
        speed = 140
        while True:
            x, sr = tts(text, speed, os.path.join(out, "vo.wav"))
            dur = len(x) / sr
            if dur <= limit or speed >= 185:
                break
            speed += 8
        x = resample_poly(x, SR, sr).astype(np.float32)
        x = hp(x, 90)
        a = int(t0 * SR)
        voice[a:a + len(x)] += x[:len(voice) - a]
        timing.append((t0, t0 + dur, text, speed))
        print(f"  VO {t0:6.1f}-{t0 + dur:6.1f}s (limit {limit:4.1f}) speed {speed}: {text[:40]}")
    ir = np.random.default_rng(2).normal(0, 1, int(0.9 * SR)) * np.exp(-np.linspace(0, 6, int(0.9 * SR)))
    wet = fftconvolve(voice, ir.astype(np.float32))[:len(voice)]
    voice = voice + 0.07 * wet / (np.abs(wet).max() + 1e-6) * np.abs(voice).max()
    voice = lp(voice, 7000)
    voice *= 0.9 / (np.abs(voice).max() + 1e-6)
    return voice.astype(np.float32), timing


def midi(m):
    return 440 * 2 ** ((m - 69) / 12)


CHORDS = [[45, 52, 57, 60, 64], [41, 48, 53, 57, 60], [48, 55, 60, 64, 67], [43, 50, 55, 59, 62]]


def pad(n):
    t = np.arange(n) / SR
    out = np.zeros((2, n), np.float32)
    bar = 8.0
    for b in range(int(DUR / bar) + 1):
        ch = CHORDS[b % 4]
        t0 = b * bar - 1.0
        s = max(0, int(t0 * SR))
        e = min(n, int((t0 + bar + 2.0) * SR))
        if s >= e:
            continue
        tt = t[s:e] - t0
        env = np.clip(tt / 2.0, 0, 1) * np.clip((bar + 2.0 - tt) / 2.0, 0, 1)
        env = env ** 1.5
        for side, det in ((0, 0.997), (1, 1.003)):
            sig = np.zeros(e - s)
            for m in ch:
                f = midi(m) * det
                for k in range(1, 6):
                    sig += np.sin(2 * np.pi * f * k * tt + k) / k ** 1.4
            out[side, s:e] += (sig * env).astype(np.float32)
    out = np.stack([lp(out[0], 1400), lp(out[1], 1400)])
    return out / np.abs(out).max()


def bells():
    out = np.zeros((2, DUR * SR), np.float32)
    rng = np.random.default_rng(8)
    scale = [57, 60, 64, 67, 69, 72, 76, 79]
    tl = np.arange(int(2.4 * SR)) / SR
    step = 0.6
    for n in range(int(DUR / step)):
        t0 = n * step
        if t0 < 12 or (t0 > 174):
            continue
        if rng.random() < 0.35:
            continue
        ch = CHORDS[int(t0 // 8) % 4]
        pool = [m + 12 for m in ch if m > 50] + [57, 60, 64, 67, 72]
        m = int(rng.choice(pool))
        f = midi(m)
        note = (np.sin(2 * np.pi * f * tl) + 0.35 * np.sin(2 * np.pi * f * 2.01 * tl) + 0.12 * np.sin(2 * np.pi * f * 3.02 * tl)) * np.exp(-tl * 2.8)
        lvl = 0.25 if t0 < 46 else 0.5
        a = int(t0 * SR)
        side = n % 2
        seg = note[:DUR * SR - a] * lvl
        out[side, a:a + len(seg)] += seg
        d = int(0.43 * SR)
        a2 = a + d
        seg2 = seg[:max(0, DUR * SR - a2)] * 0.4
        out[1 - side, a2:a2 + len(seg2)] += seg2
    return out


def pulse():
    out = np.zeros(DUR * SR, np.float32)
    tl = np.arange(int(0.5 * SR)) / SR
    for t0 in np.arange(56, 176, 1.2):
        f = 55 * np.exp(-tl * 14) + 38
        k = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tl * 7)
        a = int(t0 * SR)
        out[a:a + len(k)] += k[:DUR * SR - a] * 0.45
    return out


def ambience():
    rng = np.random.default_rng(3)
    n = DUR * SR
    t = np.arange(n) / SR
    out = np.zeros(n, np.float32)

    def window(a, b, fade=1.5):
        return np.clip((t - a) / fade, 0, 1) * np.clip((b - t) / fade, 0, 1)

    def crickets(a, b, gain):
        for carrier, rate in ((4100, 5.3), (4500, 6.1), (4850, 4.7)):
            gate = (np.sin(2 * np.pi * rate * t) > 0.55) * (np.sin(2 * np.pi * 31 * t) > 0)
            gate = lp(gate.astype(np.float32), 120)
            out[:] += (np.sin(2 * np.pi * carrier * t) * gate * window(a, b) * gain).astype(np.float32)

    crickets(0, 47, 0.025)
    crickets(86, 112, 0.04)
    crickets(166, 180, 0.03)
    wind = lp(rng.normal(0, 1, n).astype(np.float32), 500) * (0.5 + 0.5 * np.sin(t * 0.4))
    wind = wind / np.abs(wind).max()
    out += wind * 0.22 * (window(56, 86) + window(146, 166)).astype(np.float32)
    birds = np.zeros(n, np.float32)
    for t0 in np.sort(rng.uniform(58, 84, 26)):
        for k in range(rng.integers(2, 5)):
            tl = np.arange(int(0.11 * SR)) / SR
            f0 = rng.uniform(2800, 3600)
            ph = 2 * np.pi * np.cumsum(f0 + 1600 * tl / 0.11) / SR
            s = np.sin(ph) * np.sin(np.pi * tl / 0.11)
            a = int((t0 + k * 0.16) * SR)
            birds[a:a + len(s)] += s * 0.06
    out += birds
    # footsteps while the collector walks, then on the road
    for t0 in list(np.arange(57.0, 68.5, 0.46)) + list(np.arange(147, 165.5, 0.42)):
        tl = np.arange(int(0.09 * SR)) / SR
        s = lp(rng.normal(0, 1, len(tl)).astype(np.float32), 700) * np.exp(-tl * 50)
        a = int(t0 * SR)
        out[a:a + len(s)] += s * 0.22
    # light-trap lamp hum
    out += (np.sin(2 * np.pi * 100 * t) * 0.012 * window(86, 111)).astype(np.float32)
    # riser into the title + boom
    rs = hp(rng.normal(0, 1, n).astype(np.float32), 1500)
    out += rs * (np.clip((t - 40) / 6, 0, 1) ** 3 * (t < 46)) * 0.12
    tl = np.arange(int(3 * SR)) / SR
    boom = np.sin(2 * np.pi * 48 * tl) * np.exp(-tl * 1.4) * 0.8 + lp(rng.normal(0, 1, len(tl)), 300) * np.exp(-tl * 3) * 0.5
    a = int(46 * SR)
    out[a:a + len(boom)] += boom.astype(np.float32) * 0.8
    # swooshes at chapter changes
    for t0 in (56, 86, 111, 146, 166):
        tl = np.arange(int(1.8 * SR)) / SR
        sw = sosfilt(butter(2, [300, 2500], "band", fs=SR, output="sos"), rng.normal(0, 1, len(tl)))
        sw *= np.sin(np.pi * tl / 1.8) ** 2
        a = int((t0 - 0.9) * SR)
        out[a:a + len(sw)] += sw.astype(np.float32) * 0.1
    return out


def ass_time(s):
    h = int(s // 3600)
    m = int(s % 3600 // 60)
    sec = s % 60
    return "%d:%02d:%05.2f" % (h, m, sec)


def make_ass(timing, path):
    head = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,DejaVu Sans,40,&H00F2F2F2,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,160,160,24,1
Style: Logo,DejaVu Sans Mono,128,&H0070D2FF,&H000000FF,&H00201008,&H80000000,-1,0,0,0,100,100,4,0,1,5,3,5,0,0,0,1
Style: Tag,DejaVu Sans Mono,46,&H00F0F0F0,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,8,0,1,3,1,5,0,0,0,1
Style: ChapNum,DejaVu Sans Mono,34,&H0070D2FF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,10,0,1,3,1,7,150,0,150,1
Style: ChapName,DejaVu Sans Mono,84,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,6,0,1,4,2,7,150,0,190,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []

    def add(a, b, style, text):
        ev.append("Dialogue: 0,%s,%s,%s,,0,0,0,,%s" % (ass_time(a), ass_time(b), style, text))

    for t0, t1, text, _ in timing:
        sents = [s.strip() for s in text.replace("! ", "!|").replace(". ", ".|").split("|") if s.strip()]
        total = sum(len(s) for s in sents)
        cur = t0
        for s in sents:
            d = (t1 - t0) * len(s) / total
            add(cur, cur + d + 0.15, "Sub", "{\\fad(150,150)}" + s)
            cur += d
    add(47.2, 54.6, "Logo", "{\\an5\\pos(960,470)\\fad(900,900)\\blur1.5}Flora0world HUB")
    add(48.4, 54.6, "Tag", "{\\an5\\pos(960,590)\\fad(800,900)}энтомологи · природоведы · киперы")
    for (t0, num, name) in ((56.8, "ГЛАВА I", "ЛОВЛЯ"), (86.6, "ГЛАВА II", "НОЧЬ"), (111.8, "ГЛАВА III", "КАРТА"), (146.8, "ГЛАВА IV", "ПУТЕШЕСТВИЯ")):
        add(t0, t0 + 5.0, "ChapNum", "{\\fad(500,700)\\pos(150,170)}" + num)
        add(t0 + 0.2, t0 + 5.0, "ChapName", "{\\fad(600,700)\\pos(150,205)}" + name)
    add(175.6, 179.6, "Logo", "{\\an5\\pos(960,440)\\fad(900,800)\\blur1.5}Flora0world HUB")
    add(176.6, 179.6, "Tag", "{\\an5\\pos(960,560)\\fad(700,800)}Присоединяйся к сообществу")
    with open(path, "w", encoding="utf-8") as f:
        f.write(head + "\n".join(ev) + "\n")


def main(out):
    os.makedirs(out, exist_ok=True)
    print("voice...")
    voice, timing = make_voice(out)
    print("music...")
    n = DUR * SR
    p = pad(n)
    t = np.arange(n) / SR
    lvl = np.interp(t, [0, 3, 40, 46, 47, 56, 86, 111, 146, 166, 175, 180], [0, 0.55, 0.65, 0.95, 0.85, 0.85, 0.85, 0.95, 0.9, 0.85, 0.75, 0.0])
    music = p * lvl * 0.45
    music += bells() * 0.5
    pl = pulse() * 0.5
    music += np.stack([pl, pl])
    amb = ambience()
    music += np.stack([amb, amb])
    # tiny stereo decorrelation on the ambience
    music[1] = np.roll(music[1], 400)
    env = lp(np.abs(voice), 6)
    env = np.clip(env / (env.max() + 1e-6) * 3.5, 0, 1)
    duck = 1 - 0.45 * env
    mix = music * duck + np.stack([voice, voice]) * 0.95
    mix *= 0.92 / np.abs(mix).max()
    mix *= np.clip(np.minimum(t / 1.0, (DUR - t) / 2.0), 0, 1)
    pcm = (np.clip(mix, -1, 1).T * 32767).astype(np.int16)
    with wave.open(os.path.join(out, "audio.wav"), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    make_ass(timing, os.path.join(out, "subs.ass"))
    with open(os.path.join(out, "voiceover_script.txt"), "w", encoding="utf-8") as f:
        for t0, t1, text, _ in timing:
            f.write("[%s - %s] %s\n" % (ass_time(t0)[:-3], ass_time(t1)[:-3], text))
    print("audio ok")
