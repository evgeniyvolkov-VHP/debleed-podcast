#!/usr/bin/env python3
"""
Debleed Podcast 1.0 — mic bleed cleanup for DaVinci Resolve Studio.

For podcasts recorded with a separate microphone per person (2 or 3 speakers). Each mic also picks up
the other people at a lower level, which sounds like an echo. The script cuts every speaker's clip on the
timeline into pieces and disables the pieces where that track only carries someone else's echo.
Nothing is rendered: the source files stay the same and any disabled piece can be enabled again (key D).

How it tells a quiet "uh-huh" from echo: the echo on track B is a room-filtered copy of voice A,
so its spectrum can be predicted from track A. Whatever track B has that voice A and room noise
cannot explain (a different pitch, different formants) is treated as B's own voice.

Requirements: DaVinci Resolve Studio (tested on version 21), Python 3 (64-bit),
              pip install numpy soundfile
Install:  copy this file to the Resolve scripts folder
  Windows: %APPDATA%\\Blackmagic Design\\DaVinci Resolve\\Support\\Fusion\\Scripts\\Utility\\
  macOS:   ~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Utility/
Run:      Workspace → Scripts → Debleed Podcast
License:  MIT
"""
import os
import sys
import types
import warnings

try:  # Resolve не всегда видит библиотеки, поставленные через pip --user
    import site
    _user_site = site.getusersitepackages()
    if _user_site not in sys.path:
        sys.path.append(_user_site)
except Exception:
    pass
warnings.filterwarnings("ignore", message=".*sub-interpreter.*")

HOP = 0.01  # шаг анализа, 10 мс
G_BLOCK = 6000  # окраску эха оцениваем отдельно для каждой минуты записи
SIMILAR = 2.0   # выключено (2 = никогда);  порог похожести спектра на соседа, выше которого кадр считается эхом
EXPLAIN_SURE_DB = 6.0  # насколько выше порога должна быть спектральная оценка, чтобы правило громкости её не трогало
DUP_SIMILAR = 2.0  # выключено пока;  дубляж: дорожка во времени повторяет соседа...
DUP_MARGIN_DB = -0.5  # ...и дорожка тише соседа: из двух одинаковых звуков оригинал громче...
DUP_RANGE_DB = 5.0    # ...но не больше чем на столько: тихое эхо и «ага» поверх него разбирает основная проверка


# =====================================================================
#  Анализ звука
# =====================================================================

def _np():
    import numpy as np
    return np


def spectra(path, fmin=100, fmax=4000):
    """Спектр мощности в дБ (float16) каждые 10 мс, полоса голоса."""
    np = _np()
    import soundfile as sf
    info = sf.info(path)
    sr = info.samplerate
    hop = int(round(sr * HOP))
    nfft = 1 << int(np.ceil(np.log2(0.043 * sr)))
    win = np.hanning(nfft).astype(np.float32)
    k0, k1 = int(fmin * nfft / sr), int(fmax * nfft / sr) + 1
    out = []

    def emit(x):
        n = (len(x) - nfft) // hop + 1
        if n <= 0:
            return x
        idx = np.arange(nfft)[None, :] + hop * np.arange(n)[:, None]
        S = np.fft.rfft(x[idx] * win, axis=1)[:, k0:k1]
        out.append((10 * np.log10(np.abs(S) ** 2 + 1e-12)).astype(np.float16))
        return x[n * hop:]

    rest = np.zeros(nfft // 2, np.float32)  # кадр t центрирован на t*10 мс
    for block in sf.blocks(path, blocksize=hop * 1000, dtype="float32", always_2d=True):
        rest = emit(np.concatenate([rest, block.mean(axis=1)]))
    emit(np.concatenate([rest, np.zeros(nfft, np.float32)]))
    return np.concatenate(out) if out else np.full((1, k1 - k0), -120, np.float16)


def broadband(P):
    np = _np()
    out = np.empty(len(P), np.float32)
    for a in range(0, len(P), 50000):
        out[a:a + 50000] = 10 * np.log10((10 ** (P[a:a + 50000].astype(np.float32) / 10)).sum(axis=1) + 1e-12)
    return out


def smear(P, decay_db=1.5):
    """Огибающая с хвостом реверберации: спадает не быстрее decay_db за 10 мс."""
    np = _np()
    out = np.empty(P.shape, np.float16)
    cur = np.full(P.shape[1], -200.0, np.float32)
    for t in range(P.shape[0]):
        cur = np.maximum(P[t].astype(np.float32), cur - decay_db)
        out[t] = cur
    return out


def sample_rows(mask, cap=20000):
    np = _np()
    idx = np.flatnonzero(mask)
    if len(idx) > cap:
        idx = np.random.default_rng(0).choice(idx, cap, replace=False)
    return idx


def band_levels(P, n_bands=16):
    """Уровни в 16 полосах частот (дБ) каждые 10 мс."""
    np = _np()
    k = P.shape[1]
    edges = np.unique(np.round(np.geomspace(1, k, n_bands + 1)).astype(int) - 1)
    out = np.empty((P.shape[0], len(edges) - 1), np.float32)
    for a in range(0, P.shape[0], 20000):
        lin = 10 ** (P[a:a + 20000].astype(np.float32) / 10)
        for b in range(len(edges) - 1):
            out[a:a + 20000, b] = 10 * np.log10(lin[:, edges[b]:max(edges[b + 1], edges[b] + 1)].sum(axis=1) + 1e-12)
    return out


def profile_similarity(B_dst, B_src, win=50):
    """
    Насколько дорожка dst повторяет во времени соседа (0..1): корреляция того, как в каждой полосе частот
    нарастают и спадают слоги, в окне полсекунды. Эхо повторяет ритм соседа, свой голос живёт своим ритмом.
    """
    np = _np()
    n = B_dst.shape[0]

    def roll(x):
        c = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0, dtype=np.float64)])
        lo = np.clip(np.arange(n) - win // 2, 0, n)
        hi = np.clip(np.arange(n) + win // 2 + 1, 0, n)
        return (c[hi] - c[lo]), (hi - lo)[:, None]

    x, y = B_dst.astype(np.float64), B_src.astype(np.float64)
    sx, cnt = roll(x)
    sy, _ = roll(y)
    sxx, _ = roll(x * x)
    syy, _ = roll(y * y)
    sxy, _ = roll(x * y)
    cov = sxy - sx * sy / cnt
    vx = sxx - sx * sx / cnt
    vy = syy - sy * sy / cnt
    r = cov / np.sqrt(np.maximum(vx * vy, 1e-9))
    # полосы, где у соседа нет движения (тишина), в среднее не берём
    w = np.sqrt(np.maximum(vy, 0))
    return ((r * w).sum(axis=1) / (w.sum(axis=1) + 1e-9)).astype(np.float32)


def residual_score(P_dst, sources, valid_dst, margin_db):
    """
    дБ энергии на дорожке dst над шумом, которую нельзя объяснить эхом соседей.
    sources: [(огибающая соседа со хвостом реверберации, кадры, где сосед говорит один,
               уровни соседа в полосах частот, кадры, где сосед говорит и не тише этой дорожки)].
    Эхо всех соседей складывается, у каждого своя окраска комнаты G.
    """
    np = _np()
    n = P_dst.shape[0]
    noise = np.percentile(P_dst[sample_rows(valid_dst)].astype(np.float32), 20, axis=0)
    B = G_BLOCK
    nb = (n + B - 1) // B

    def est(S_src, rows):
        G = np.median(P_dst[rows].astype(np.float32) - S_src[rows].astype(np.float32), axis=0)
        return np.convolve(G, np.ones(5) / 5, mode="same")

    Gs = []  # для каждого соседа: окраска эха по блокам в минуту
    sims = []
    for S_src, calib, P_src, act_src in sources:
        rows = sample_rows(calib)
        G_all = est(S_src, rows) if len(rows) >= 200 else np.full(P_dst.shape[1], -15.0, np.float32)
        # кадры, где дорожка по профилю повторяет соседа: это эхо при любой громкости
        if SIMILAR <= 1:
            sim = profile_similarity(band_levels(P_dst), P_src)
            sims.append(sim)
            calib = calib | (act_src & (sim > SIMILAR))
        blocks = []
        for k in range(nb):
            lo, hi = max(0, (k - 1) * B), min(n, (k + 2) * B)  # окно три минуты вокруг блока
            r = np.flatnonzero(calib[lo:hi]) + lo
            if len(r) >= 300:
                if len(r) > 5000:
                    r = np.random.default_rng(k).choice(r, 5000, replace=False)
                # эхо в этом месте записи может быть сильнее среднего (люди пересели, говорили громче),
                # поэтому берём местную оценку, но не даём ей стать заметно слабее общей
                # и не громче самого голоса соседа: такое бывает только если это чужой голос, а не эхо
                blocks.append(np.minimum(np.maximum(est(S_src, r), G_all - 3), -1.0))
            else:
                blocks.append(G_all)
        Gs.append((G_all, blocks))
    lin = lambda d: 10 ** (d / 10)
    nz_total = lin(noise).sum()
    nz3 = lin(noise + 3)[None, :]
    score = np.empty(n, np.float32)
    for k in range(nb):
        a, e = k * B, min(n, (k + 1) * B)
        pd = lin(P_dst[a:e].astype(np.float32))
        pred = nz3.copy()
        for (S_src, _, _, _), (_, blocks) in zip(sources, Gs):
            pred = pred + lin(S_src[a:e].astype(np.float32) + blocks[k][None, :] + margin_db)
        score[a:e] = 10 * np.log10(np.maximum(pd - pred, 0).sum(axis=1) / nz_total + 1e-6)
    leaks = [float(np.median(G_all)) for G_all, _ in Gs]
    block_leaks = [[round(float(np.median(g)), 1) for g in blocks] for _, blocks in Gs]
    residual_score.last_sims = sims
    return np.convolve(score, np.ones(5) / 5, mode="same"), leaks, block_leaks


def runs(mask):
    np = _np()
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.where(d == 1)[0], np.where(d == -1)[0]


def drop_short(mask, min_len, value=True):
    m = mask.copy()
    s, e = runs(m if value else ~m)
    for a, b in zip(s, e):
        if b - a < min_len:
            m[a:b] = not value
    return m


def dilate(mask, pre, post):
    out = mask.copy()
    s, e = runs(mask)
    for a, b in zip(s, e):
        out[max(0, a - pre): min(len(mask), b + post)] = True
    return out


def default_params():
    return types.SimpleNamespace(margin=11.0, thresh=14.0, vad=12.0, pre=80, hold=250, min_open=100, min_gap=300)


def analyze_tracks(tracks, p, log=print, tr=None):
    """
    tracks: [(исходный файл, где на таймлайне начинается файл в секундах)], 2 или 3 спикера.
    Возвращает:
      masks  — для каждого файла маска своего голоса в сетке 10 мс от начала файла,
      quiet  — маска «тихих реплик»: свой голос, пока говорит кто-то другой,
      overlaps — [(сек таймлайна, длительность, сколько говорят одновременно)].
    """
    np = _np()
    tr = tr or Tr("en")
    N = len(tracks)
    P = []
    for i, (src, _) in enumerate(tracks):
        log(tr("spectrum", n=i + 1))
        P.append(spectra(src))
    offs = [int(round(t / HOP)) for _, t in tracks]
    base = min(offs)
    L = max(o + len(x) for o, x in zip(offs, P)) - base
    starts = [o - base for o in offs]
    lens = [len(x) for x in P]
    g, v = [], []
    for i in range(N):
        gi = np.full((L, P[i].shape[1]), -120, np.float16)
        gi[starts[i]:starts[i] + lens[i]] = P[i]
        vi = np.zeros(L, bool)
        vi[starts[i]:starts[i] + lens[i]] = True
        g.append(gi)
        v.append(vi)
        P[i] = None
    if not np.logical_and.reduce(v).any():
        raise UserError("no_overlap")

    d = [broadband(x) for x in g]
    act = [v[i] & (d[i] > np.percentile(d[i][v[i]], 10) + p.vad) for i in range(N)]
    # «говорит один»: громче всех остальных минимум на 3 дБ
    solo = []
    for j in range(N):
        others = np.max(np.stack([d[k] for k in range(N) if k != j]), axis=0)
        solo.append(act[j] & (d[j] - others > 3))

    log(tr("comparing"))
    S = [smear(x) for x in g]
    bands = [band_levels(x) for x in g] if DUP_SIMILAR <= 1 or SIMILAR <= 1 else [None] * N
    scores = []
    for i in range(N):
        # эхом по профилю считаем только кадры, где сосед не тише этой дорожки:
        # когда говорит сам человек, его голос эхом звучит у соседа тише, и сюда не попадёт
        srcs = [(S[j], solo[j] & v[i], bands[j], act[j] & v[i] & (d[j] - d[i] > 0)) for j in range(N) if j != i]
        sc, leaks, block_leaks = residual_score(g[i], srcs, v[i], p.margin)
        scores.append(sc)
        if getattr(p, "stats", None) is not None:
            p.stats[f"leak_by_minute_track{i + 1}"] = block_leaks
            others = [j for j in range(N) if j != i]
            for j, sim in zip(others, residual_score.last_sims if SIMILAR <= 1 else []):
                own_i = solo[i] & v[i]
                echo_j = solo[j] & v[i]
                p.stats[f"similarity_mic{i + 1}_to_voice{j + 1}"] = {
                    "own_speech": np.percentile(sim[own_i], [50, 90, 99]).round(2).tolist() if own_i.any() else None,
                    "echo_only": np.percentile(sim[echo_j], [10, 50, 90]).round(2).tolist() if echo_j.any() else None,
                    "share_marked_echo": float((sim[act[j] & v[i]] > SIMILAR).mean()) if (act[j] & v[i]).any() else None,
                }
        names = [str(j + 1) for j in range(N) if j != i]
        log(tr("mic_echo", i=i + 1, parts=", ".join(tr("echo_part", n=n, db=f"{-l:.0f}") for n, l in zip(names, leaks))))
    # Правило громкости: пока говорят другие, дорожка почти не громче их ожидаемого эха —
    # значит, её громкость целиком объясняется эхом, это дубляж. Порог связан с чувствительностью.
    # чувствительность 5 → 4 дБ, 6 → 2 дБ, 7 и выше → правило выключено (сохраняются самые тихие «ага»)
    explain_db = p.thresh - 10.0
    explained_share = []
    for i in (range(N) if explain_db >= 2.0 else []):
        pred = np.full(L, -200.0, np.float32)
        others_act = np.zeros(L, bool)
        for j in range(N):
            if j == i:
                continue
            cal = solo[j] & v[i]
            gb = float(np.median((d[i] - d[j])[cal])) if cal.sum() > 200 else -20.0
            pred = 10 * np.log10(10 ** (pred / 10) + 10 ** ((d[j] + gb) / 10))
            others_act |= act[j]
        # спектральная оценка при этом должна быть неуверенной: уверенный другой голос не трогаем
        explained = v[i] & others_act & (d[i] - pred < explain_db) & (scores[i] < p.thresh + EXPLAIN_SURE_DB)
        scores[i] = np.where(explained, np.minimum(scores[i], -20.0), scores[i])
        explained_share.append(float(explained[v[i]].mean()))
    if getattr(p, "stats", None) is not None:
        p.stats["explained_by_echo_share"] = explained_share
        p.stats["explain_db"] = explain_db

    # Прямое правило дубляжа: дорожка по профилю повторяет соседа и звучит тише его — это эхо,
    # даже если по громкости оно почти как сам голос (люди сидят близко, говорят громко)
    dup_share = []
    for i in (range(N) if DUP_SIMILAR <= 1 else []):
        dup = np.zeros(L, bool)
        for j in range(N):
            if j == i:
                continue
            sim = profile_similarity(bands[i], bands[j])
            diff = d[i] - d[j]
            dup |= act[j] & v[i] & (sim > DUP_SIMILAR) & (diff < DUP_MARGIN_DB) & (diff > -DUP_RANGE_DB)
        scores[i] = np.where(dup, np.minimum(scores[i], -20.0), scores[i])
        dup_share.append(float(dup[v[i]].mean()))
    if getattr(p, "stats", None) is not None:
        p.stats["duplicate_share"] = dup_share
    del S, g, bands

    stats = getattr(p, "stats", None)
    if stats is not None:
        # распределение оценки там, где человек явно говорит сам, и там, где говорит только сосед
        pct = [5, 10, 25, 50, 75, 90, 95, 99]
        for i in range(N):
            others_solo = np.zeros(L, bool)
            for j in range(N):
                if j != i:
                    others_solo |= solo[j]
            own = solo[i] & v[i]
            echo = others_solo & v[i] & ~act[i]
            echo_loud = others_solo & v[i]
            silent = v[i] & ~np.logical_or.reduce(act)
            stats[f"track{i + 1}"] = {
                "frames_own": int(own.sum()), "frames_echo": int(echo.sum()), "frames_silent": int(silent.sum()),
                "score_own": np.percentile(scores[i][own], pct).round(1).tolist() if own.any() else None,
                "score_echo_only": np.percentile(scores[i][echo_loud], pct).round(1).tolist() if echo_loud.any() else None,
                "score_silence": np.percentile(scores[i][silent], pct).round(1).tolist() if silent.any() else None,
                "level_own_db": np.percentile(d[i][own], [10, 50, 90]).round(1).tolist() if own.any() else None,
                "level_echo_db": np.percentile(d[i][echo_loud], [10, 50, 90]).round(1).tolist() if echo_loud.any() else None,
                "level_floor_db": float(np.percentile(d[i][v[i]], 10)),
                "open_share_raw": float((scores[i][v[i]] > p.thresh).mean()),
                "open_share_first5min_raw": float((scores[i][v[i]][:30000] > p.thresh).mean()),
            }
        stats["open_raw_by_minute"] = {f"track{i + 1}": [round(float((scores[i][k:k + 6000] > p.thresh).mean()), 2)
                                                         for k in range(0, L, 6000)] for i in range(N)}
        # насколько громкость дорожки объясняется одним эхом соседей (дБ над ожидаемым эхом)
        for i in range(N):
            pred = np.full(L, -200.0, np.float32)
            for j in range(N):
                if j == i:
                    continue
                cal = solo[j] & v[i]
                gb = float(np.median((d[i] - d[j])[cal])) if cal.sum() > 200 else -20.0
                pred = 10 * np.log10(10 ** (pred / 10) + 10 ** ((d[j] + gb) / 10))
            excess = d[i] - pred
            opened = v[i] & (scores[i] > p.thresh)
            others_act = np.zeros(L, bool)
            for j in range(N):
                if j != i:
                    others_act |= act[j]
            sel = opened & others_act
            first = np.zeros(L, bool)
            first[:18000] = True
            stats[f"excess_over_echo_track{i + 1}"] = {
                "opened_while_others_talk_all": np.percentile(excess[sel], [10, 25, 50, 75, 90]).round(1).tolist() if sel.any() else None,
                "opened_while_others_talk_first3min": np.percentile(excess[sel & first], [10, 25, 50, 75, 90]).round(1).tolist() if (sel & first).any() else None,
                "share_opened_first3min_explained_by_echo_lt4db": float((excess[sel & first] < 4).mean()) if (sel & first).any() else None,
                "share_opened_all_explained_by_echo_lt4db": float((excess[sel] < 4).mean()) if sel.any() else None,
                "own_solo_excess": np.percentile(excess[solo[i] & v[i]], [5, 10, 50]).round(1).tolist() if (solo[i] & v[i]).any() else None,
            }
        stats["percentiles"] = pct
        stats["thresh"] = p.thresh
        stats["margin"] = p.margin

    f = lambda ms: max(1, int(round(ms / 1000 / HOP)))
    speech = []
    for i in range(N):
        m = drop_short(v[i] & (scores[i] > p.thresh), f(p.min_open))
        m = dilate(m, f(p.pre), f(p.hold))
        speech.append(drop_short(m, f(p.min_gap), value=False))

    count = np.sum(np.stack(speech), axis=0)
    masks, quiet = [], []
    for i in range(N):
        others_talk = (count - speech[i]) > 0
        q = speech[i] & others_talk
        masks.append(speech[i][starts[i]:starts[i] + lens[i]])
        quiet.append(q[starts[i]:starts[i] + lens[i]])

    overlap = drop_short(count >= 2, f(500))
    s, e = runs(overlap)
    overlaps = [((base + a) * HOP, (b - a) * HOP, int(count[a:b].max())) for a, b in zip(s, e)]
    return masks, quiet, overlaps


def clip_pieces(mask, quiet, left, dur, fps, min_off_frames):
    """
    Делит клип (in-точка left, длина dur в кадрах таймлайна) на куски.
    Возвращает [(смещение от начала клипа, длина, включён, тихая_реплика)].
    """
    np = _np()
    n = len(mask)
    cm = np.concatenate([[0], np.cumsum(mask)])
    cq = np.concatenate([[0], np.cumsum(quiet)])
    fr = np.arange(dur)
    a = np.clip(np.floor((left + fr) / fps / HOP).astype(int), 0, n)
    b = np.clip(np.ceil((left + fr + 1) / fps / HOP).astype(int), 0, n)
    keep = (cm[b] - cm[a]) > 0  # кадр включён, если в нём есть хоть кусочек своего голоса
    keep = drop_short(keep, max(1, min_off_frames), value=False)
    pieces = []
    s, e = runs(keep)
    pos = 0
    for x, y in zip(s, e):
        x, y = int(x), int(y)  # Resolve понимает только обычные int, numpy-числа он молча игнорирует
        if x > pos:
            pieces.append((pos, x - pos, False, "off"))
        qa, qb = a[x], b[y - 1]
        share = (cq[qb] - cq[qa]) / max(1, cm[qb] - cm[qa])  # доля куска, где говорит кто-то ещё
        if (y - x) < 2 * fps and share > 0.3:
            kind = "reply"      # короткая реплика поверх чужой речи
        elif share > 0.3:
            kind = "overlap"    # длинный кусок одновременной речи
        else:
            kind = "own"
        pieces.append((x, y - x, True, kind))
        pos = y
    if pos < dur:
        pieces.append((pos, int(dur) - pos, False, "off"))
    return [(int(o), int(n), bool(on), kind) for o, n, on, kind in pieces]


# =====================================================================
#  Interface languages / Языки интерфейса
# =====================================================================

LANGS = [("en", "English"), ("es", "Español"), ("ru", "Русский"), ("pt", "Português"), ("fr", "Français"),
         ("de", "Deutsch"), ("zh", "中文"), ("ja", "日本語"), ("hi", "हिन्दी"), ("ar", "العربية")]

TEXTS = {
    "en": {
        "language": "Language", "refresh": "Refresh", "mode": "Mode", "mode2": "2 speakers", "mode3": "3 speakers",
        "sec_tracks": "Speaker tracks", "speaker": "Speaker {n}", "speaker_to": "Speaker {n} →",
        "sec_where": "Where to cut", "sec_settings": "Settings",
        "strictness": "Echo strictness", "sensitivity": "Sensitivity",
        "hint": "Higher strictness removes more echo but may clip a reply. Higher sensitivity keeps quieter "
                "backchannels (“uh-huh”) but leaves more echo. At 5–6 echo-level duplicates are removed firmly, "
                "from 7 up the quietest backchannels are kept.",
        "hold": "Tail after a phrase, ms", "minoff": "Don't mute pieces shorter than, ms",
        "backup": "Duplicate the timeline first",
        "colors": "Color pieces: echo orange, backchannels yellow, crosstalk pink",
        "debug": "Debug: write a detailed log to debleed_run_debug.json",
        "run": "Cut", "analyze": "Analyze only", "close": "Close",
        "same_track": "Same track (cut in place)", "new_track": "New track (original muted)", "db": "{v} dB",
        "tl_info": "<b>Timeline:</b> {name} · {fps} fps · audio tracks: {n}", "tl_none": "<b>No timeline open</b>",
        "backup_name": "{name} before Debleed {time}",
        "spectrum": "Analyzing track {n}…", "comparing": "Comparing spectra…",
        "no_overlap": "Clips on the selected tracks don't overlap in time.",
        "mic_echo": "Mic {i}: echo of {parts}", "echo_part": "voice {n} is {db} dB quieter",
        "less3": "The timeline has fewer than three audio tracks.",
        "less2": "The timeline has fewer than two audio tracks.",
        "no_clips": "Track A{t} has no audio clips.",
        "no_file": "Couldn't find the source file of the clip on A{t}: {path}",
        "other_files": "A{t} has {n} clip(s) from other files, leaving them alone.",
        "no_value": "Resolve didn't return {what}.", "what_left": "the clip's start in the file",
        "what_start": "the clip position", "what_dur": "the clip length",
        "what_piece_start": "the piece position", "what_piece_dur": "the piece length",
        "restored": "The original clip on A{t} is back in place.",
        "restore_failed": "Couldn't restore the clip on A{t}. Open the “before Debleed” timeline copy or press Ctrl+Z.",
        "not_accepted": "Resolve rejected the clip on A{t}. Check that the track isn't locked and matches the "
                        "channel type (mono/stereo).",
        "misplaced": "Resolve placed a piece in the wrong spot. Cutting stopped.",
        "misplaced_batch": "Resolve placed pieces on A{t} incorrectly (batch {b}). Cutting stopped.",
        "markers_failed": "Couldn't remove old markers: {e}",
        "merge_denied": "Resolve didn't allow re-joining pieces on A{t}.",
        "merge_failed": "Couldn't re-join pieces on A{t}. Open the “before Debleed” timeline copy.",
        "merged": "A{t}: re-joined {n} pieces from the previous cut.",
        "debug_log": "Debug log: {path}", "debug_failed": "Couldn't write the debug log: {e}",
        "missing_lib": "Library {name} is missing. Install it into the Python that Resolve uses: "
                       "pip install numpy soundfile, then restart Resolve.",
        "open_tl": "Open a timeline and press “{run}” again.",
        "other_tl": "Another timeline is open: {name}. Track lists were updated, check the selection and press "
                    "“{run}” again.",
        "markers_removed": "Removed old Debleed markers: {n}.",
        "distinct": "Choose a different track for each speaker.",
        "same_file": "Two speakers use the same source file. Check the track selection.",
        "analyze_done": "Analysis only: nothing was cut.",
        "backup_ok": "Timeline copy created.", "backup_fail": "Couldn't duplicate the timeline, continuing.",
        "cut_denied": "Resolve didn't allow cutting the clip on A{t}. Is the track locked?",
        "track_done": "A{t}: {n} pieces, {off} muted.",
        "done": "Done. Select a muted piece and press D to enable it again.",
        "error": "Error: {e}",
    },
    "ru": {
        "language": "Язык", "refresh": "Обновить", "mode": "Режим", "mode2": "2 спикера", "mode3": "3 спикера",
        "sec_tracks": "Дорожки спикеров", "speaker": "Спикер {n}", "speaker_to": "Спикер {n} →",
        "sec_where": "Где резать", "sec_settings": "Настройки",
        "strictness": "Строгость к эху", "sensitivity": "Чувствительность",
        "hint": "Строгость выше: больше эха уходит, но риск срезать реплику. Чувствительность выше: ловит более "
                "тихие «ага», но чаще оставляет эхо. На 5–6 дубляж на уровне эха убирается жёстко, с 7 сохраняются "
                "самые тихие «ага».",
        "hold": "Запас после фразы, мс", "minoff": "Не выключать куски короче, мс",
        "backup": "Сначала сделать копию таймлайна",
        "colors": "Красить куски: эхо оранжевым, «ага» жёлтым, одновременная речь розовым",
        "debug": "Отладка: писать подробный лог в debleed_run_debug.json",
        "run": "Нарезать", "analyze": "Только анализ", "close": "Закрыть",
        "same_track": "Та же дорожка (клип режется на месте)", "new_track": "Новая дорожка (оригинал выключается)",
        "db": "{v} дБ",
        "tl_info": "<b>Таймлайн:</b> {name} · {fps} fps · аудиодорожек: {n}", "tl_none": "<b>Таймлайн не открыт</b>",
        "backup_name": "{name} до Debleed {time}",
        "spectrum": "Спектр дорожки {n}…", "comparing": "Сравниваю спектры…",
        "no_overlap": "Клипы на выбранных дорожках не пересекаются по времени.",
        "mic_echo": "Микрофон {i}: эхо {parts}", "echo_part": "голоса {n} тише на {db} дБ",
        "less3": "На таймлайне меньше трёх аудиодорожек.", "less2": "На таймлайне меньше двух аудиодорожек.",
        "no_clips": "На дорожке A{t} нет аудиоклипов.",
        "no_file": "Не нашёл исходный файл клипа на A{t}: {path}",
        "other_files": "На A{t} есть {n} клип(ов) из других файлов, их не трогаю.",
        "no_value": "Resolve не вернул {what}.", "what_left": "начало клипа в файле", "what_start": "позицию клипа",
        "what_dur": "длину клипа", "what_piece_start": "позицию куска", "what_piece_dur": "длину куска",
        "restored": "Исходный клип на A{t} возвращён на место.",
        "restore_failed": "Не удалось вернуть клип на A{t}. Откройте копию таймлайна «до Debleed» или нажмите Ctrl+Z.",
        "not_accepted": "Resolve не принял клип на A{t}. Проверьте, что дорожка не заблокирована и совпадает "
                        "по типу (моно/стерео).",
        "misplaced": "Resolve положил кусок не туда, куда нужно. Нарезка остановлена.",
        "misplaced_batch": "На A{t} Resolve положил куски неправильно (пачка {b}). Нарезка остановлена.",
        "markers_failed": "Старые маркеры убрать не удалось: {e}",
        "merge_denied": "Resolve не дал собрать куски на A{t}.",
        "merge_failed": "Не удалось собрать куски на A{t}. Откройте копию таймлайна «до Debleed».",
        "merged": "A{t}: собрал обратно {n} кусков прошлой нарезки.",
        "debug_log": "Лог отладки: {path}", "debug_failed": "Не удалось записать лог отладки: {e}",
        "missing_lib": "Не хватает библиотеки {name}. Установите в Python, который видит Resolve: "
                       "pip install numpy soundfile, затем перезапустите Resolve.",
        "open_tl": "Откройте таймлайн и нажмите «{run}» ещё раз.",
        "other_tl": "Открыт другой таймлайн: {name}. Списки дорожек обновлены, проверьте выбор и нажмите "
                    "«{run}» ещё раз.",
        "markers_removed": "Убрано старых маркеров Debleed: {n}.",
        "distinct": "Выберите разные дорожки для каждого спикера.",
        "same_file": "У двух спикеров один и тот же исходный файл. Проверьте выбор дорожек.",
        "analyze_done": "Только анализ: нарезка не выполнялась.",
        "backup_ok": "Копия таймлайна сделана.", "backup_fail": "Копию таймлайна сделать не удалось, продолжаю.",
        "cut_denied": "Resolve не дал разрезать клип на A{t}. Дорожка заблокирована?",
        "track_done": "A{t}: {n} кусков, выключено {off}.",
        "done": "Готово. Выключенный кусок включается обратно клавишей D.", "error": "Ошибка: {e}",
    },
    "es": {
        "language": "Idioma", "refresh": "Actualizar", "mode": "Modo", "mode2": "2 personas", "mode3": "3 personas",
        "sec_tracks": "Pistas de cada persona", "speaker": "Persona {n}", "speaker_to": "Persona {n} →",
        "sec_where": "Dónde cortar", "sec_settings": "Ajustes",
        "strictness": "Rigor con el eco", "sensitivity": "Sensibilidad",
        "hint": "Más rigor quita más eco, pero puede cortar una réplica. Más sensibilidad conserva los “ajá” más "
                "bajos, pero deja más eco. En 5–6 el duplicado al nivel del eco se elimina con firmeza; desde 7 se "
                "conservan los “ajá” más bajos.",
        "hold": "Margen tras la frase, ms", "minoff": "No silenciar trozos más cortos que, ms",
        "backup": "Duplicar la línea de tiempo antes",
        "colors": "Colorear trozos: eco naranja, “ajá” amarillo, voces simultáneas rosa",
        "debug": "Depuración: guardar registro detallado en debleed_run_debug.json",
        "run": "Cortar", "analyze": "Solo analizar", "close": "Cerrar",
        "same_track": "Misma pista (se corta en su sitio)", "new_track": "Pista nueva (el original se desactiva)",
        "db": "{v} dB",
        "tl_info": "<b>Línea de tiempo:</b> {name} · {fps} fps · pistas de audio: {n}",
        "tl_none": "<b>No hay línea de tiempo abierta</b>",
        "backup_name": "{name} antes de Debleed {time}",
        "spectrum": "Analizando pista {n}…", "comparing": "Comparando espectros…",
        "no_overlap": "Los clips de las pistas elegidas no coinciden en el tiempo.",
        "mic_echo": "Micrófono {i}: eco de {parts}", "echo_part": "la voz {n}, {db} dB más bajo",
        "less3": "La línea de tiempo tiene menos de tres pistas de audio.",
        "less2": "La línea de tiempo tiene menos de dos pistas de audio.",
        "no_clips": "La pista A{t} no tiene clips de audio.",
        "no_file": "No encuentro el archivo original del clip en A{t}: {path}",
        "other_files": "A{t} tiene {n} clip(s) de otros archivos, no los toco.",
        "no_value": "Resolve no devolvió {what}.", "what_left": "el inicio del clip en el archivo",
        "what_start": "la posición del clip", "what_dur": "la duración del clip",
        "what_piece_start": "la posición del trozo", "what_piece_dur": "la duración del trozo",
        "restored": "El clip original de A{t} ha vuelto a su sitio.",
        "restore_failed": "No se pudo devolver el clip a A{t}. Abre la copia “antes de Debleed” o pulsa Ctrl+Z.",
        "not_accepted": "Resolve no aceptó el clip en A{t}. Comprueba que la pista no esté bloqueada y que el tipo "
                        "coincida (mono/estéreo).",
        "misplaced": "Resolve colocó un trozo en el sitio equivocado. Corte detenido.",
        "misplaced_batch": "Resolve colocó mal los trozos en A{t} (lote {b}). Corte detenido.",
        "markers_failed": "No se pudieron quitar los marcadores antiguos: {e}",
        "merge_denied": "Resolve no permitió unir los trozos en A{t}.",
        "merge_failed": "No se pudieron unir los trozos en A{t}. Abre la copia “antes de Debleed”.",
        "merged": "A{t}: unidos {n} trozos del corte anterior.",
        "debug_log": "Registro de depuración: {path}", "debug_failed": "No se pudo guardar el registro: {e}",
        "missing_lib": "Falta la biblioteca {name}. Instálala en el Python que usa Resolve: "
                       "pip install numpy soundfile, y reinicia Resolve.",
        "open_tl": "Abre una línea de tiempo y pulsa “{run}” otra vez.",
        "other_tl": "Hay otra línea de tiempo abierta: {name}. Listas de pistas actualizadas; revisa la selección "
                    "y pulsa “{run}” otra vez.",
        "markers_removed": "Marcadores antiguos de Debleed eliminados: {n}.",
        "distinct": "Elige una pista distinta para cada persona.",
        "same_file": "Dos personas usan el mismo archivo. Revisa la selección de pistas.",
        "analyze_done": "Solo análisis: no se cortó nada.",
        "backup_ok": "Copia de la línea de tiempo creada.",
        "backup_fail": "No se pudo duplicar la línea de tiempo, sigo.",
        "cut_denied": "Resolve no permitió cortar el clip en A{t}. ¿Está bloqueada la pista?",
        "track_done": "A{t}: {n} trozos, {off} desactivados.",
        "done": "Listo. Selecciona un trozo desactivado y pulsa D para activarlo.", "error": "Error: {e}",
    },
    "pt": {
        "language": "Idioma", "refresh": "Atualizar", "mode": "Modo", "mode2": "2 pessoas", "mode3": "3 pessoas",
        "sec_tracks": "Faixas de cada pessoa", "speaker": "Pessoa {n}", "speaker_to": "Pessoa {n} →",
        "sec_where": "Onde cortar", "sec_settings": "Ajustes",
        "strictness": "Rigor com o eco", "sensitivity": "Sensibilidade",
        "hint": "Mais rigor remove mais eco, mas pode cortar uma réplica. Mais sensibilidade mantém os “aham” mais "
                "baixos, mas deixa mais eco. Em 5–6 o duplicado no nível do eco é removido com firmeza; a partir de "
                "7 os “aham” mais baixos são mantidos.",
        "hold": "Folga após a frase, ms", "minoff": "Não silenciar trechos menores que, ms",
        "backup": "Duplicar a linha do tempo antes",
        "colors": "Colorir trechos: eco laranja, “aham” amarelo, falas simultâneas rosa",
        "debug": "Depuração: gravar log detalhado em debleed_run_debug.json",
        "run": "Cortar", "analyze": "Só analisar", "close": "Fechar",
        "same_track": "Mesma faixa (corta no lugar)", "new_track": "Faixa nova (original desativado)",
        "db": "{v} dB",
        "tl_info": "<b>Linha do tempo:</b> {name} · {fps} fps · faixas de áudio: {n}",
        "tl_none": "<b>Nenhuma linha do tempo aberta</b>",
        "backup_name": "{name} antes do Debleed {time}",
        "spectrum": "Analisando faixa {n}…", "comparing": "Comparando espectros…",
        "no_overlap": "Os clipes das faixas escolhidas não se sobrepõem no tempo.",
        "mic_echo": "Microfone {i}: eco de {parts}", "echo_part": "voz {n}, {db} dB mais baixo",
        "less3": "A linha do tempo tem menos de três faixas de áudio.",
        "less2": "A linha do tempo tem menos de duas faixas de áudio.",
        "no_clips": "A faixa A{t} não tem clipes de áudio.",
        "no_file": "Não encontrei o arquivo original do clipe em A{t}: {path}",
        "other_files": "A{t} tem {n} clipe(s) de outros arquivos, não mexo neles.",
        "no_value": "O Resolve não retornou {what}.", "what_left": "o início do clipe no arquivo",
        "what_start": "a posição do clipe", "what_dur": "a duração do clipe",
        "what_piece_start": "a posição do trecho", "what_piece_dur": "a duração do trecho",
        "restored": "O clipe original em A{t} voltou ao lugar.",
        "restore_failed": "Não foi possível devolver o clipe a A{t}. Abra a cópia “antes do Debleed” ou "
                          "pressione Ctrl+Z.",
        "not_accepted": "O Resolve recusou o clipe em A{t}. Verifique se a faixa não está bloqueada e se o tipo "
                        "confere (mono/estéreo).",
        "misplaced": "O Resolve colocou um trecho no lugar errado. Corte interrompido.",
        "misplaced_batch": "O Resolve colocou trechos errados em A{t} (lote {b}). Corte interrompido.",
        "markers_failed": "Não foi possível remover os marcadores antigos: {e}",
        "merge_denied": "O Resolve não permitiu juntar os trechos em A{t}.",
        "merge_failed": "Não foi possível juntar os trechos em A{t}. Abra a cópia “antes do Debleed”.",
        "merged": "A{t}: juntei {n} trechos do corte anterior.",
        "debug_log": "Log de depuração: {path}", "debug_failed": "Não foi possível gravar o log: {e}",
        "missing_lib": "Falta a biblioteca {name}. Instale no Python que o Resolve usa: "
                       "pip install numpy soundfile, e reinicie o Resolve.",
        "open_tl": "Abra uma linha do tempo e pressione “{run}” de novo.",
        "other_tl": "Outra linha do tempo está aberta: {name}. Listas de faixas atualizadas; confira a seleção e "
                    "pressione “{run}” de novo.",
        "markers_removed": "Marcadores antigos do Debleed removidos: {n}.",
        "distinct": "Escolha uma faixa diferente para cada pessoa.",
        "same_file": "Duas pessoas usam o mesmo arquivo. Confira a seleção de faixas.",
        "analyze_done": "Só análise: nada foi cortado.",
        "backup_ok": "Cópia da linha do tempo criada.",
        "backup_fail": "Não foi possível duplicar a linha do tempo, continuando.",
        "cut_denied": "O Resolve não permitiu cortar o clipe em A{t}. A faixa está bloqueada?",
        "track_done": "A{t}: {n} trechos, {off} desativados.",
        "done": "Pronto. Selecione um trecho desativado e pressione D para reativá-lo.", "error": "Erro: {e}",
    },
    "fr": {
        "language": "Langue", "refresh": "Actualiser", "mode": "Mode", "mode2": "2 intervenants",
        "mode3": "3 intervenants", "sec_tracks": "Pistes des intervenants", "speaker": "Intervenant {n}",
        "speaker_to": "Intervenant {n} →", "sec_where": "Où couper", "sec_settings": "Réglages",
        "strictness": "Rigueur envers l'écho", "sensitivity": "Sensibilité",
        "hint": "Plus de rigueur retire plus d'écho, mais peut couper une réplique. Plus de sensibilité garde les "
                "« mm-hmm » les plus faibles, mais laisse plus d'écho. À 5–6 le doublon au niveau de l'écho est "
                "retiré fermement ; à partir de 7 les « mm-hmm » les plus faibles sont conservés.",
        "hold": "Marge après une phrase, ms", "minoff": "Ne pas couper les morceaux plus courts que, ms",
        "backup": "Dupliquer la timeline d'abord",
        "colors": "Colorer les morceaux : écho orange, « mm-hmm » jaune, paroles simultanées rose",
        "debug": "Débogage : écrire un journal détaillé dans debleed_run_debug.json",
        "run": "Découper", "analyze": "Analyser seulement", "close": "Fermer",
        "same_track": "Même piste (découpe sur place)", "new_track": "Nouvelle piste (original désactivé)",
        "db": "{v} dB",
        "tl_info": "<b>Timeline :</b> {name} · {fps} i/s · pistes audio : {n}",
        "tl_none": "<b>Aucune timeline ouverte</b>",
        "backup_name": "{name} avant Debleed {time}",
        "spectrum": "Analyse de la piste {n}…", "comparing": "Comparaison des spectres…",
        "no_overlap": "Les clips des pistes choisies ne se chevauchent pas dans le temps.",
        "mic_echo": "Micro {i} : écho de {parts}", "echo_part": "la voix {n}, {db} dB plus bas",
        "less3": "La timeline a moins de trois pistes audio.", "less2": "La timeline a moins de deux pistes audio.",
        "no_clips": "La piste A{t} n'a pas de clips audio.",
        "no_file": "Fichier source du clip introuvable sur A{t} : {path}",
        "other_files": "A{t} contient {n} clip(s) d'autres fichiers, je n'y touche pas.",
        "no_value": "Resolve n'a pas renvoyé {what}.", "what_left": "le début du clip dans le fichier",
        "what_start": "la position du clip", "what_dur": "la durée du clip",
        "what_piece_start": "la position du morceau", "what_piece_dur": "la durée du morceau",
        "restored": "Le clip d'origine sur A{t} est remis en place.",
        "restore_failed": "Impossible de remettre le clip sur A{t}. Ouvrez la copie « avant Debleed » ou "
                          "appuyez sur Ctrl+Z.",
        "not_accepted": "Resolve a refusé le clip sur A{t}. Vérifiez que la piste n'est pas verrouillée et que le "
                        "type correspond (mono/stéréo).",
        "misplaced": "Resolve a placé un morceau au mauvais endroit. Découpe arrêtée.",
        "misplaced_batch": "Resolve a mal placé les morceaux sur A{t} (lot {b}). Découpe arrêtée.",
        "markers_failed": "Impossible de retirer les anciens marqueurs : {e}",
        "merge_denied": "Resolve n'a pas permis de réassembler les morceaux sur A{t}.",
        "merge_failed": "Impossible de réassembler les morceaux sur A{t}. Ouvrez la copie « avant Debleed ».",
        "merged": "A{t} : {n} morceaux de la découpe précédente réassemblés.",
        "debug_log": "Journal de débogage : {path}", "debug_failed": "Impossible d'écrire le journal : {e}",
        "missing_lib": "La bibliothèque {name} manque. Installez-la dans le Python utilisé par Resolve : "
                       "pip install numpy soundfile, puis redémarrez Resolve.",
        "open_tl": "Ouvrez une timeline et appuyez à nouveau sur « {run} ».",
        "other_tl": "Une autre timeline est ouverte : {name}. Listes de pistes mises à jour ; vérifiez la sélection "
                    "et appuyez à nouveau sur « {run} ».",
        "markers_removed": "Anciens marqueurs Debleed retirés : {n}.",
        "distinct": "Choisissez une piste différente pour chaque intervenant.",
        "same_file": "Deux intervenants utilisent le même fichier. Vérifiez le choix des pistes.",
        "analyze_done": "Analyse seulement : rien n'a été découpé.",
        "backup_ok": "Copie de la timeline créée.", "backup_fail": "Impossible de dupliquer la timeline, je continue.",
        "cut_denied": "Resolve n'a pas permis de couper le clip sur A{t}. La piste est-elle verrouillée ?",
        "track_done": "A{t} : {n} morceaux, {off} désactivés.",
        "done": "Terminé. Sélectionnez un morceau désactivé et appuyez sur D pour le réactiver.",
        "error": "Erreur : {e}",
    },
    "de": {
        "language": "Sprache", "refresh": "Aktualisieren", "mode": "Modus", "mode2": "2 Sprecher",
        "mode3": "3 Sprecher", "sec_tracks": "Spuren der Sprecher", "speaker": "Sprecher {n}",
        "speaker_to": "Sprecher {n} →", "sec_where": "Wo schneiden", "sec_settings": "Einstellungen",
        "strictness": "Strenge beim Echo", "sensitivity": "Empfindlichkeit",
        "hint": "Mehr Strenge entfernt mehr Echo, kann aber eine Erwiderung abschneiden. Mehr Empfindlichkeit behält "
                "leisere „mhm“, lässt aber mehr Echo stehen. Bei 5–6 werden Doppelungen auf Echo-Niveau konsequent "
                "entfernt, ab 7 bleiben die leisesten „mhm“ erhalten.",
        "hold": "Nachlauf nach einem Satz, ms", "minoff": "Stücke kürzer als ... nicht stummschalten, ms",
        "backup": "Zuerst die Timeline duplizieren",
        "colors": "Stücke einfärben: Echo orange, „mhm“ gelb, gleichzeitiges Sprechen rosa",
        "debug": "Debug: ausführliches Log in debleed_run_debug.json schreiben",
        "run": "Schneiden", "analyze": "Nur analysieren", "close": "Schließen",
        "same_track": "Gleiche Spur (vor Ort schneiden)", "new_track": "Neue Spur (Original deaktiviert)",
        "db": "{v} dB",
        "tl_info": "<b>Timeline:</b> {name} · {fps} fps · Audiospuren: {n}", "tl_none": "<b>Keine Timeline offen</b>",
        "backup_name": "{name} vor Debleed {time}",
        "spectrum": "Analysiere Spur {n}…", "comparing": "Vergleiche Spektren…",
        "no_overlap": "Die Clips der gewählten Spuren überschneiden sich zeitlich nicht.",
        "mic_echo": "Mikrofon {i}: Echo von {parts}", "echo_part": "Stimme {n}, {db} dB leiser",
        "less3": "Die Timeline hat weniger als drei Audiospuren.",
        "less2": "Die Timeline hat weniger als zwei Audiospuren.",
        "no_clips": "Spur A{t} hat keine Audioclips.",
        "no_file": "Quelldatei des Clips auf A{t} nicht gefunden: {path}",
        "other_files": "A{t} enthält {n} Clip(s) aus anderen Dateien, die bleiben unverändert.",
        "no_value": "Resolve hat {what} nicht geliefert.", "what_left": "den Clipanfang in der Datei",
        "what_start": "die Clipposition", "what_dur": "die Cliplänge",
        "what_piece_start": "die Position des Stücks", "what_piece_dur": "die Länge des Stücks",
        "restored": "Der ursprüngliche Clip auf A{t} ist wieder an seinem Platz.",
        "restore_failed": "Clip auf A{t} konnte nicht wiederhergestellt werden. Öffnen Sie die Kopie „vor Debleed“ "
                          "oder drücken Sie Strg+Z.",
        "not_accepted": "Resolve hat den Clip auf A{t} abgelehnt. Prüfen Sie, ob die Spur gesperrt ist und der Typ "
                        "passt (Mono/Stereo).",
        "misplaced": "Resolve hat ein Stück an die falsche Stelle gelegt. Schnitt abgebrochen.",
        "misplaced_batch": "Resolve hat Stücke auf A{t} falsch gelegt (Paket {b}). Schnitt abgebrochen.",
        "markers_failed": "Alte Marker konnten nicht entfernt werden: {e}",
        "merge_denied": "Resolve hat das Zusammenfügen der Stücke auf A{t} nicht erlaubt.",
        "merge_failed": "Stücke auf A{t} konnten nicht zusammengefügt werden. Öffnen Sie die Kopie „vor Debleed“.",
        "merged": "A{t}: {n} Stücke des letzten Schnitts wieder zusammengefügt.",
        "debug_log": "Debug-Log: {path}", "debug_failed": "Debug-Log konnte nicht geschrieben werden: {e}",
        "missing_lib": "Bibliothek {name} fehlt. Installieren Sie sie in das Python, das Resolve nutzt: "
                       "pip install numpy soundfile, dann Resolve neu starten.",
        "open_tl": "Öffnen Sie eine Timeline und drücken Sie erneut „{run}“.",
        "other_tl": "Eine andere Timeline ist offen: {name}. Spurlisten aktualisiert, Auswahl prüfen und erneut "
                    "„{run}“ drücken.",
        "markers_removed": "Alte Debleed-Marker entfernt: {n}.",
        "distinct": "Wählen Sie für jeden Sprecher eine andere Spur.",
        "same_file": "Zwei Sprecher nutzen dieselbe Datei. Prüfen Sie die Spurauswahl.",
        "analyze_done": "Nur Analyse: nichts wurde geschnitten.",
        "backup_ok": "Kopie der Timeline erstellt.",
        "backup_fail": "Timeline konnte nicht dupliziert werden, mache weiter.",
        "cut_denied": "Resolve hat das Schneiden des Clips auf A{t} nicht erlaubt. Ist die Spur gesperrt?",
        "track_done": "A{t}: {n} Stücke, {off} deaktiviert.",
        "done": "Fertig. Deaktiviertes Stück auswählen und D drücken, um es wieder zu aktivieren.",
        "error": "Fehler: {e}",
    },
    "zh": {
        "language": "语言", "refresh": "刷新", "mode": "模式", "mode2": "2 位说话人", "mode3": "3 位说话人",
        "sec_tracks": "说话人轨道", "speaker": "说话人 {n}", "speaker_to": "说话人 {n} →",
        "sec_where": "在哪里剪切", "sec_settings": "设置",
        "strictness": "回声严格度", "sensitivity": "灵敏度",
        "hint": "严格度越高，去掉的回声越多，但可能剪掉一句回应。灵敏度越高，越能保留轻声的“嗯”，但会留下更多回声。"
                "在 5–6 时会坚决去掉回声级别的重复声；从 7 开始保留最轻的“嗯”。",
        "hold": "句尾保留时长，毫秒", "minoff": "短于此长度的片段不静音，毫秒",
        "backup": "先复制时间线",
        "colors": "片段着色：回声橙色，“嗯”黄色，同时说话粉色",
        "debug": "调试：将详细日志写入 debleed_run_debug.json",
        "run": "剪切", "analyze": "仅分析", "close": "关闭",
        "same_track": "同一轨道（原地剪切）", "new_track": "新轨道（原片段禁用）", "db": "{v} dB",
        "tl_info": "<b>时间线：</b>{name} · {fps} fps · 音频轨道：{n}", "tl_none": "<b>未打开时间线</b>",
        "backup_name": "{name} Debleed 之前 {time}",
        "spectrum": "正在分析轨道 {n}…", "comparing": "正在比较频谱…",
        "no_overlap": "所选轨道上的片段在时间上没有重叠。",
        "mic_echo": "麦克风 {i}：{parts}", "echo_part": "说话人 {n} 的回声低 {db} dB",
        "less3": "时间线上的音频轨道少于三条。", "less2": "时间线上的音频轨道少于两条。",
        "no_clips": "轨道 A{t} 上没有音频片段。",
        "no_file": "找不到 A{t} 上片段的源文件：{path}",
        "other_files": "A{t} 上有 {n} 个来自其他文件的片段，不做改动。",
        "no_value": "Resolve 没有返回{what}。", "what_left": "片段在文件中的起点", "what_start": "片段位置",
        "what_dur": "片段长度", "what_piece_start": "小段位置", "what_piece_dur": "小段长度",
        "restored": "A{t} 上的原始片段已放回原处。",
        "restore_failed": "无法恢复 A{t} 上的片段。请打开“Debleed 之前”的时间线副本，或按 Ctrl+Z。",
        "not_accepted": "Resolve 拒绝了 A{t} 上的片段。请检查轨道未锁定且类型一致（单声道/立体声）。",
        "misplaced": "Resolve 把一个小段放错了位置。剪切已停止。",
        "misplaced_batch": "Resolve 在 A{t} 上放错了小段（第 {b} 批）。剪切已停止。",
        "markers_failed": "无法删除旧标记：{e}",
        "merge_denied": "Resolve 不允许在 A{t} 上合并小段。",
        "merge_failed": "无法合并 A{t} 上的小段。请打开“Debleed 之前”的时间线副本。",
        "merged": "A{t}：已合并上次剪切的 {n} 个小段。",
        "debug_log": "调试日志：{path}", "debug_failed": "无法写入调试日志：{e}",
        "missing_lib": "缺少库 {name}。请在 Resolve 使用的 Python 中安装：pip install numpy soundfile，然后重启 Resolve。",
        "open_tl": "请打开一条时间线，然后再按一次“{run}”。",
        "other_tl": "当前打开的是另一条时间线：{name}。轨道列表已更新，请检查选择后再按一次“{run}”。",
        "markers_removed": "已删除旧的 Debleed 标记：{n} 个。",
        "distinct": "请为每位说话人选择不同的轨道。",
        "same_file": "两位说话人使用了同一个源文件。请检查轨道选择。",
        "analyze_done": "仅分析：未进行剪切。",
        "backup_ok": "已创建时间线副本。", "backup_fail": "无法复制时间线，继续执行。",
        "cut_denied": "Resolve 不允许剪切 A{t} 上的片段。轨道是否已锁定？",
        "track_done": "A{t}：{n} 个小段，禁用 {off} 个。",
        "done": "完成。选中已禁用的小段并按 D 即可重新启用。", "error": "错误：{e}",
    },
    "ja": {
        "language": "言語", "refresh": "更新", "mode": "モード", "mode2": "話者 2 人", "mode3": "話者 3 人",
        "sec_tracks": "話者のトラック", "speaker": "話者 {n}", "speaker_to": "話者 {n} →",
        "sec_where": "カットする場所", "sec_settings": "設定",
        "strictness": "エコーへの厳しさ", "sensitivity": "感度",
        "hint": "厳しさを上げるとエコーをより多く消せますが、相づちを切ることがあります。感度を上げると小さな「うん」を"
                "残せますが、エコーも残りやすくなります。5–6 ではエコーと同じ音量の重複をしっかり消し、7 以上では最も"
                "小さな「うん」も残します。",
        "hold": "フレーズ後の余白（ミリ秒）", "minoff": "これより短い部分はミュートしない（ミリ秒）",
        "backup": "先にタイムラインを複製する",
        "colors": "部分に色を付ける：エコーはオレンジ、「うん」は黄色、同時の発話はピンク",
        "debug": "デバッグ：詳細ログを debleed_run_debug.json に書き出す",
        "run": "カット", "analyze": "分析のみ", "close": "閉じる",
        "same_track": "同じトラック（その場でカット）", "new_track": "新しいトラック（元は無効化）", "db": "{v} dB",
        "tl_info": "<b>タイムライン：</b>{name} · {fps} fps · オーディオトラック：{n}",
        "tl_none": "<b>タイムラインが開いていません</b>",
        "backup_name": "{name} Debleed 前 {time}",
        "spectrum": "トラック {n} を分析中…", "comparing": "スペクトルを比較中…",
        "no_overlap": "選択したトラックのクリップが時間的に重なっていません。",
        "mic_echo": "マイク {i}：{parts}", "echo_part": "話者 {n} のエコーは {db} dB 小さい",
        "less3": "タイムラインのオーディオトラックが 3 本未満です。",
        "less2": "タイムラインのオーディオトラックが 2 本未満です。",
        "no_clips": "トラック A{t} にオーディオクリップがありません。",
        "no_file": "A{t} のクリップの元ファイルが見つかりません：{path}",
        "other_files": "A{t} に他のファイルのクリップが {n} 個あります。触りません。",
        "no_value": "Resolve が{what}を返しませんでした。", "what_left": "ファイル内のクリップ開始位置",
        "what_start": "クリップの位置", "what_dur": "クリップの長さ",
        "what_piece_start": "部分の位置", "what_piece_dur": "部分の長さ",
        "restored": "A{t} の元のクリップを元の位置に戻しました。",
        "restore_failed": "A{t} のクリップを戻せませんでした。「Debleed 前」のタイムラインを開くか Ctrl+Z を押してください。",
        "not_accepted": "Resolve が A{t} のクリップを受け付けませんでした。トラックがロックされていないこと、種類"
                        "（モノ/ステレオ）が一致することを確認してください。",
        "misplaced": "Resolve が部分を違う位置に置きました。カットを中止しました。",
        "misplaced_batch": "Resolve が A{t} の部分を正しく置けませんでした（バッチ {b}）。カットを中止しました。",
        "markers_failed": "古いマーカーを削除できませんでした：{e}",
        "merge_denied": "Resolve が A{t} の部分の結合を許可しませんでした。",
        "merge_failed": "A{t} の部分を結合できませんでした。「Debleed 前」のタイムラインを開いてください。",
        "merged": "A{t}：前回のカットの {n} 個の部分を結合しました。",
        "debug_log": "デバッグログ：{path}", "debug_failed": "デバッグログを書き出せませんでした：{e}",
        "missing_lib": "ライブラリ {name} がありません。Resolve が使う Python にインストールしてください："
                       "pip install numpy soundfile のあと Resolve を再起動。",
        "open_tl": "タイムラインを開いて、もう一度「{run}」を押してください。",
        "other_tl": "別のタイムラインが開いています：{name}。トラック一覧を更新しました。選択を確認して、もう一度"
                    "「{run}」を押してください。",
        "markers_removed": "古い Debleed マーカーを削除しました：{n} 個。",
        "distinct": "話者ごとに違うトラックを選んでください。",
        "same_file": "2 人の話者が同じ元ファイルを使っています。トラックの選択を確認してください。",
        "analyze_done": "分析のみ：カットは行っていません。",
        "backup_ok": "タイムラインを複製しました。", "backup_fail": "タイムラインを複製できませんでした。続行します。",
        "cut_denied": "Resolve が A{t} のクリップのカットを許可しませんでした。トラックはロックされていませんか？",
        "track_done": "A{t}：{n} 個の部分、{off} 個を無効化。",
        "done": "完了。無効な部分を選んで D を押すと再び有効になります。", "error": "エラー：{e}",
    },
    "hi": {
        "language": "भाषा", "refresh": "ताज़ा करें", "mode": "मोड", "mode2": "2 वक्ता", "mode3": "3 वक्ता",
        "sec_tracks": "वक्ताओं के ट्रैक", "speaker": "वक्ता {n}", "speaker_to": "वक्ता {n} →",
        "sec_where": "कहाँ काटें", "sec_settings": "सेटिंग्स",
        "strictness": "गूँज पर सख़्ती", "sensitivity": "संवेदनशीलता",
        "hint": "सख़्ती ज़्यादा होने पर गूँज ज़्यादा हटती है, पर कोई जवाब कट सकता है। संवेदनशीलता ज़्यादा होने पर धीमे "
                "“हाँ” बचे रहते हैं, पर गूँज भी ज़्यादा रहती है। 5–6 पर गूँज के स्तर की दोहराई आवाज़ सख़्ती से हटती है, "
                "7 से ऊपर सबसे धीमे “हाँ” भी बचे रहते हैं।",
        "hold": "वाक्य के बाद का अंतराल, ms", "minoff": "इससे छोटे टुकड़े बंद न करें, ms",
        "backup": "पहले टाइमलाइन की कॉपी बनाएँ",
        "colors": "टुकड़ों को रंग दें: गूँज नारंगी, “हाँ” पीला, साथ में बोलना गुलाबी",
        "debug": "डिबग: विस्तृत लॉग debleed_run_debug.json में लिखें",
        "run": "काटें", "analyze": "केवल विश्लेषण", "close": "बंद करें",
        "same_track": "वही ट्रैक (उसी जगह काटें)", "new_track": "नया ट्रैक (मूल बंद)", "db": "{v} dB",
        "tl_info": "<b>टाइमलाइन:</b> {name} · {fps} fps · ऑडियो ट्रैक: {n}", "tl_none": "<b>कोई टाइमलाइन खुली नहीं</b>",
        "backup_name": "{name} Debleed से पहले {time}",
        "spectrum": "ट्रैक {n} का विश्लेषण…", "comparing": "स्पेक्ट्रम की तुलना…",
        "no_overlap": "चुने गए ट्रैकों के क्लिप समय में एक-दूसरे से नहीं मिलते।",
        "mic_echo": "माइक {i}: {parts}", "echo_part": "वक्ता {n} की गूँज {db} dB धीमी",
        "less3": "टाइमलाइन में तीन से कम ऑडियो ट्रैक हैं।", "less2": "टाइमलाइन में दो से कम ऑडियो ट्रैक हैं।",
        "no_clips": "ट्रैक A{t} पर कोई ऑडियो क्लिप नहीं है।",
        "no_file": "A{t} पर क्लिप की मूल फ़ाइल नहीं मिली: {path}",
        "other_files": "A{t} पर दूसरी फ़ाइलों के {n} क्लिप हैं, उन्हें नहीं छू रहा।",
        "no_value": "Resolve ने {what} नहीं लौटाया।", "what_left": "फ़ाइल में क्लिप की शुरुआत",
        "what_start": "क्लिप की जगह", "what_dur": "क्लिप की लंबाई",
        "what_piece_start": "टुकड़े की जगह", "what_piece_dur": "टुकड़े की लंबाई",
        "restored": "A{t} पर मूल क्लिप वापस अपनी जगह पर है।",
        "restore_failed": "A{t} पर क्लिप वापस नहीं रखा जा सका। “Debleed से पहले” वाली कॉपी खोलें या Ctrl+Z दबाएँ।",
        "not_accepted": "Resolve ने A{t} पर क्लिप नहीं लिया। जाँचें कि ट्रैक लॉक नहीं है और प्रकार (मोनो/स्टीरियो) मेल खाता है।",
        "misplaced": "Resolve ने टुकड़ा गलत जगह रख दिया। कटिंग रोक दी गई।",
        "misplaced_batch": "Resolve ने A{t} पर टुकड़े गलत रखे (बैच {b})। कटिंग रोक दी गई।",
        "markers_failed": "पुराने मार्कर नहीं हटे: {e}",
        "merge_denied": "Resolve ने A{t} पर टुकड़े जोड़ने नहीं दिए।",
        "merge_failed": "A{t} पर टुकड़े नहीं जुड़े। “Debleed से पहले” वाली कॉपी खोलें।",
        "merged": "A{t}: पिछली कटिंग के {n} टुकड़े वापस जोड़े।",
        "debug_log": "डिबग लॉग: {path}", "debug_failed": "डिबग लॉग नहीं लिखा जा सका: {e}",
        "missing_lib": "लाइब्रेरी {name} नहीं है। Resolve जिस Python का उपयोग करता है उसमें इंस्टॉल करें: "
                       "pip install numpy soundfile, फिर Resolve दोबारा चालू करें।",
        "open_tl": "कोई टाइमलाइन खोलें और फिर से “{run}” दबाएँ।",
        "other_tl": "दूसरी टाइमलाइन खुली है: {name}। ट्रैक सूची अपडेट हो गई, चुनाव जाँचें और फिर से “{run}” दबाएँ।",
        "markers_removed": "पुराने Debleed मार्कर हटाए: {n}।",
        "distinct": "हर वक्ता के लिए अलग ट्रैक चुनें।",
        "same_file": "दो वक्ता एक ही मूल फ़ाइल का उपयोग कर रहे हैं। ट्रैक का चुनाव जाँचें।",
        "analyze_done": "केवल विश्लेषण: कुछ नहीं काटा गया।",
        "backup_ok": "टाइमलाइन की कॉपी बन गई।", "backup_fail": "टाइमलाइन की कॉपी नहीं बनी, आगे बढ़ रहा हूँ।",
        "cut_denied": "Resolve ने A{t} पर क्लिप काटने नहीं दिया। क्या ट्रैक लॉक है?",
        "track_done": "A{t}: {n} टुकड़े, {off} बंद।",
        "done": "हो गया। बंद टुकड़ा चुनकर D दबाएँ, वह फिर से चालू हो जाएगा।", "error": "त्रुटि: {e}",
    },
    "ar": {
        "language": "اللغة", "refresh": "تحديث", "mode": "الوضع", "mode2": "متحدثان", "mode3": "3 متحدثين",
        "sec_tracks": "مسارات المتحدثين", "speaker": "المتحدث {n}", "speaker_to": "المتحدث {n} ←",
        "sec_where": "أين يتم القص", "sec_settings": "الإعدادات",
        "strictness": "الصرامة مع الصدى", "sensitivity": "الحساسية",
        "hint": "صرامة أعلى تزيل صدى أكثر لكنها قد تقطع ردًا قصيرًا. حساسية أعلى تحتفظ بعبارات «آها» الخافتة لكنها "
                "تترك صدى أكثر. عند 5–6 يُزال التكرار بمستوى الصدى بحزم، ومن 7 فما فوق تبقى أخفت عبارات «آها».",
        "hold": "مهلة بعد الجملة، ملّي ثانية", "minoff": "لا تكتم المقاطع الأقصر من، ملّي ثانية",
        "backup": "انسخ الخط الزمني أولًا",
        "colors": "تلوين المقاطع: الصدى برتقالي، «آها» أصفر، الكلام المتزامن وردي",
        "debug": "تصحيح الأخطاء: كتابة سجل مفصل في debleed_run_debug.json",
        "run": "قص", "analyze": "تحليل فقط", "close": "إغلاق",
        "same_track": "المسار نفسه (القص في مكانه)", "new_track": "مسار جديد (يُعطَّل الأصل)", "db": "{v} dB",
        "tl_info": "<b>الخط الزمني:</b> {name} · {fps} fps · مسارات الصوت: {n}", "tl_none": "<b>لا يوجد خط زمني مفتوح</b>",
        "backup_name": "{name} قبل Debleed {time}",
        "spectrum": "جارٍ تحليل المسار {n}…", "comparing": "جارٍ مقارنة الأطياف…",
        "no_overlap": "مقاطع المسارات المختارة لا تتداخل زمنيًا.",
        "mic_echo": "الميكروفون {i}: {parts}", "echo_part": "صدى المتحدث {n} أخفض بـ {db} dB",
        "less3": "في الخط الزمني أقل من ثلاثة مسارات صوت.", "less2": "في الخط الزمني أقل من مساري صوت.",
        "no_clips": "لا توجد مقاطع صوت على المسار A{t}.",
        "no_file": "لم أجد الملف الأصلي للمقطع على A{t}: {path}",
        "other_files": "على A{t} يوجد {n} مقطع من ملفات أخرى، لن ألمسها.",
        "no_value": "لم يُرجع Resolve {what}.", "what_left": "بداية المقطع في الملف", "what_start": "موضع المقطع",
        "what_dur": "طول المقطع", "what_piece_start": "موضع الجزء", "what_piece_dur": "طول الجزء",
        "restored": "أُعيد المقطع الأصلي على A{t} إلى مكانه.",
        "restore_failed": "تعذّرت إعادة المقطع إلى A{t}. افتح نسخة «قبل Debleed» أو اضغط Ctrl+Z.",
        "not_accepted": "رفض Resolve المقطع على A{t}. تأكد أن المسار غير مقفل وأن النوع متطابق (أحادي/ستيريو).",
        "misplaced": "وضع Resolve جزءًا في مكان خاطئ. توقف القص.",
        "misplaced_batch": "وضع Resolve الأجزاء على A{t} بشكل خاطئ (الدفعة {b}). توقف القص.",
        "markers_failed": "تعذّر حذف العلامات القديمة: {e}",
        "merge_denied": "لم يسمح Resolve بدمج الأجزاء على A{t}.",
        "merge_failed": "تعذّر دمج الأجزاء على A{t}. افتح نسخة «قبل Debleed».",
        "merged": "A{t}: دُمج {n} جزءًا من القص السابق.",
        "debug_log": "سجل التصحيح: {path}", "debug_failed": "تعذّرت كتابة سجل التصحيح: {e}",
        "missing_lib": "المكتبة {name} غير موجودة. ثبّتها في Python الذي يستخدمه Resolve: "
                       "pip install numpy soundfile، ثم أعد تشغيل Resolve.",
        "open_tl": "افتح خطًا زمنيًا واضغط «{run}» مرة أخرى.",
        "other_tl": "خط زمني آخر مفتوح: {name}. تم تحديث قوائم المسارات، راجع الاختيار واضغط «{run}» مرة أخرى.",
        "markers_removed": "حُذفت علامات Debleed القديمة: {n}.",
        "distinct": "اختر مسارًا مختلفًا لكل متحدث.",
        "same_file": "متحدثان يستخدمان الملف الأصلي نفسه. راجع اختيار المسارات.",
        "analyze_done": "تحليل فقط: لم يتم أي قص.",
        "backup_ok": "أُنشئت نسخة من الخط الزمني.", "backup_fail": "تعذّر نسخ الخط الزمني، أتابع.",
        "cut_denied": "لم يسمح Resolve بقص المقطع على A{t}. هل المسار مقفل؟",
        "track_done": "A{t}: {n} جزءًا، عُطّل {off}.",
        "done": "تم. حدّد جزءًا معطّلًا واضغط D لتفعيله مجددًا.", "error": "خطأ: {e}",
    },
}

# Настройки и лог отладки: Windows — %APPDATA%\DebleedPodcast, macOS — ~/Library/Application Support/DebleedPodcast
if os.environ.get("APPDATA"):
    _SETTINGS_DIR = os.path.join(os.environ["APPDATA"], "DebleedPodcast")
elif sys.platform == "darwin":
    _SETTINGS_DIR = os.path.expanduser("~/Library/Application Support/DebleedPodcast")
else:
    _SETTINGS_DIR = os.path.expanduser("~/.config/DebleedPodcast")
SETTINGS_PATH = os.path.join(_SETTINGS_DIR, "settings.json")


def load_settings():
    try:
        import json
        with open(SETTINGS_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def save_settings(data):
    try:
        import json
        os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
        with open(SETTINGS_PATH, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
    except Exception:
        pass


def default_lang():
    saved = load_settings().get("lang")
    if saved in TEXTS:
        return saved
    try:
        import locale
        code = (locale.getlocale()[0] or "")[:2].lower()
    except Exception:
        code = ""
    return code if code in TEXTS else "en"


class Tr:
    """Текущий язык интерфейса: tr("key", n=1) → строка на выбранном языке (если перевода нет — английская)."""

    def __init__(self, lang):
        self.lang = lang

    def __call__(self, key, **kw):
        s = TEXTS.get(self.lang, {}).get(key) or TEXTS["en"].get(key, key)
        return s.format(**kw) if kw else s


class UserError(RuntimeError):
    """Ошибка с ключом перевода: текст подставляется на языке интерфейса в момент показа."""

    def __init__(self, key, **kw):
        super().__init__(key)
        self.key, self.kw = key, kw


# =====================================================================
#  DaVinci Resolve
# =====================================================================

PIECE_COLORS = {"off": "Orange", "reply": "Yellow", "overlap": "Pink"}
MARKER_NOTE = "Debleed: переслушать"  # такие маркеры ставили прошлые версии, их убираем


def resolve_main(resolve, bmd):
    fu = resolve.Fusion()
    ui = fu.UIManager
    disp = bmd.UIDispatcher(ui)
    tr = Tr(default_lang())

    # Таймлайн не запоминается при запуске: перед каждым действием берётся тот, что открыт сейчас
    proj = tl = None
    fps = 25.0
    tracks = []
    built_for = {"id": None}  # для какого таймлайна заполнены списки дорожек

    def tl_id(t):
        try:
            return t.GetUniqueId()
        except Exception:
            return t.GetName()

    def refresh():
        nonlocal proj, tl, fps, tracks
        proj = resolve.GetProjectManager().GetCurrentProject()
        tl = proj.GetCurrentTimeline() if proj else None
        if not tl:
            tracks = []
            return False
        try:
            fps = float(str(tl.GetSetting("timelineFrameRate")).split()[0])
        except Exception:
            fps = 25.0
        tracks = [(i, f"A{i}: {tl.GetTrackName('audio', i)}") for i in range(1, tl.GetTrackCount("audio") + 1)]
        return True

    def row(label_id, *w):
        return ui.HGroup({"Weight": 0}, [ui.Label({"ID": label_id, "Text": "", "MinimumSize": [215, 0], "Weight": 0}), *w])

    win = disp.AddWindow(
        {"ID": "DebleedWin", "WindowTitle": "Debleed Podcast", "Geometry": [300, 160, 680, 720]},
        ui.VGroup({"Spacing": 6}, [
            ui.HGroup({"Weight": 0}, [ui.Label({"ID": "TlName", "Text": ""}),
                                      ui.Label({"ID": "LangLbl", "Text": "", "Weight": 0, "MinimumSize": [80, 0],
                                                "Alignment": {"AlignRight": True, "AlignVCenter": True}}),
                                      ui.ComboBox({"ID": "Lang", "Weight": 0, "MinimumSize": [110, 0]}),
                                      ui.Button({"ID": "Refresh", "Text": "", "Weight": 0})]),
            row("ModeLbl", ui.ComboBox({"ID": "Mode"})),
            ui.Label({"ID": "SecTracks", "Text": "", "Weight": 0}),
            row("Src1Lbl", ui.ComboBox({"ID": "Src1"})),
            row("Src2Lbl", ui.ComboBox({"ID": "Src2"})),
            row("Src3Lbl", ui.ComboBox({"ID": "Src3"})),
            ui.Label({"ID": "SecWhere", "Text": "", "Weight": 0}),
            row("Dst1Lbl", ui.ComboBox({"ID": "Dst1"})),
            row("Dst2Lbl", ui.ComboBox({"ID": "Dst2"})),
            row("Dst3Lbl", ui.ComboBox({"ID": "Dst3"})),
            ui.Label({"ID": "SecSettings", "Text": "", "Weight": 0}),
            row("MarginLbl", ui.Slider({"ID": "Margin", "Minimum": 4, "Maximum": 18, "Value": 11}),
                ui.Label({"ID": "MarginVal", "Text": "", "MinimumSize": [50, 0], "Weight": 0})),
            row("ThreshLbl", ui.Slider({"ID": "Thresh", "Minimum": 1, "Maximum": 10, "Value": 5}),
                ui.Label({"ID": "ThreshVal", "Text": "5", "MinimumSize": [50, 0], "Weight": 0})),
            ui.Label({"ID": "Hint", "Text": "", "WordWrap": True, "Weight": 0, "MinimumSize": [0, 48],
                      "Alignment": {"AlignTop": True}}),
            row("HoldLbl", ui.SpinBox({"ID": "Hold", "Minimum": 50, "Maximum": 1000, "Value": 250, "SingleStep": 50})),
            row("MinOffLbl", ui.SpinBox({"ID": "MinOff", "Minimum": 80, "Maximum": 3000, "Value": 300,
                                         "SingleStep": 50})),
            ui.CheckBox({"ID": "Backup", "Text": "", "Checked": True, "Weight": 0}),
            ui.CheckBox({"ID": "Colors", "Text": "", "Checked": True, "Weight": 0}),
            ui.CheckBox({"ID": "Debug", "Text": "", "Checked": False, "Weight": 0}),
            ui.HGroup({"Weight": 0}, [ui.Button({"ID": "Run", "Text": "", "MinimumSize": [0, 32]}),
                                      ui.Button({"ID": "Analyze", "Text": "", "Weight": 0}),
                                      ui.Button({"ID": "Close", "Text": "", "Weight": 0})]),
            ui.TextEdit({"ID": "Log", "ReadOnly": True}),
        ]))
    itm = win.GetItems()
    for _, name in LANGS:
        itm["Lang"].AddItem(name)
    itm["Lang"].CurrentIndex = [c for c, _ in LANGS].index(tr.lang)
    itm["Mode"].AddItem(tr("mode2"))
    itm["Mode"].AddItem(tr("mode3"))

    log_lines = []

    def log(msg):
        log_lines.append(str(msg))
        itm["Log"].PlainText = "\n".join(log_lines)
        print("Debleed:", msg)

    def n_speakers():
        return 2 + itm["Mode"].CurrentIndex

    def fill_combos():
        """Заполнить списки дорожками текущего таймлайна. Выбор сохраняется, если такие дорожки есть."""
        keep = {key: itm[key].CurrentIndex for key in ("Src1", "Src2", "Src3", "Dst1", "Dst2", "Dst3")}
        same_tl = built_for["id"] == (tl_id(tl) if tl else None)
        for k in (1, 2, 3):
            src, dst = itm[f"Src{k}"], itm[f"Dst{k}"]
            src.Clear()
            dst.Clear()
            for _, label in tracks:
                src.AddItem(label)
            dst.AddItem(tr("same_track"))
            dst.AddItem(tr("new_track"))
            for _, label in tracks:
                dst.AddItem(label)
            if tracks:
                src.CurrentIndex = keep[f"Src{k}"] if same_tl and 0 <= keep[f"Src{k}"] < len(tracks) \
                    else min(k - 1, len(tracks) - 1)
                dst.CurrentIndex = keep[f"Dst{k}"] if same_tl and 0 <= keep[f"Dst{k}"] < len(tracks) + 2 else 0
        built_for["id"] = tl_id(tl) if tl else None
        if tl:
            itm["TlName"].Text = tr("tl_info", name=tl.GetName(), fps=f"{fps:g}", n=len(tracks))
        else:
            itm["TlName"].Text = tr("tl_none")
        on_mode(None)

    def on_mode(ev):
        on = n_speakers() == 3
        if on and len(tracks) < 3:
            itm["Mode"].CurrentIndex = 0
            on = False
            if ev is not None:
                log(tr("less3"))
        itm["Src3"].Enabled = on
        itm["Dst3"].Enabled = on

    def on_refresh(ev):
        refresh()
        fill_combos()

    def apply_texts():
        """Все подписи окна на выбранном языке."""
        set_text = {
            "LangLbl": tr("language"), "Refresh": tr("refresh"), "ModeLbl": tr("mode"),
            "SecTracks": f"<b>{tr('sec_tracks')}</b>", "SecWhere": f"<b>{tr('sec_where')}</b>",
            "SecSettings": f"<b>{tr('sec_settings')}</b>",
            "MarginLbl": tr("strictness"), "ThreshLbl": tr("sensitivity"), "Hint": f"<small>{tr('hint')}</small>",
            "HoldLbl": tr("hold"), "MinOffLbl": tr("minoff"), "Backup": tr("backup"), "Colors": tr("colors"),
            "Debug": tr("debug"), "Run": tr("run"), "Analyze": tr("analyze"), "Close": tr("close"),
            "MarginVal": tr("db", v=itm["Margin"].Value),
        }
        for k in (1, 2, 3):
            set_text[f"Src{k}Lbl"] = tr("speaker", n=k)
            set_text[f"Dst{k}Lbl"] = tr("speaker_to", n=k)
        for key, text in set_text.items():
            itm[key].Text = text
        mode = itm["Mode"].CurrentIndex
        itm["Mode"].Clear()
        itm["Mode"].AddItem(tr("mode2"))
        itm["Mode"].AddItem(tr("mode3"))
        itm["Mode"].CurrentIndex = max(0, mode)
        fill_combos()  # подписи «та же дорожка / новая дорожка» и строка таймлайна

    def on_lang(ev):
        idx = itm["Lang"].CurrentIndex
        if 0 <= idx < len(LANGS) and LANGS[idx][0] != tr.lang:
            tr.lang = LANGS[idx][0]
            settings = load_settings()
            settings["lang"] = tr.lang
            save_settings(settings)
            apply_texts()

    refresh()
    apply_texts()

    def on_margin(ev):
        itm["MarginVal"].Text = tr("db", v=itm["Margin"].Value)

    def on_thresh(ev):
        itm["ThreshVal"].Text = str(itm["Thresh"].Value)

    def clip_path(it):
        mpi = it.GetMediaPoolItem()
        return mpi.GetClipProperty("File Path") if mpi else None

    def main_clip(track):
        items = [it for it in (tl.GetItemListInTrack("audio", track) or []) if clip_path(it)]
        if not items:
            raise UserError("no_clips", t=track)
        it = max(items, key=lambda x: x.GetDuration())
        path = clip_path(it)
        if not os.path.isfile(path):
            raise UserError("no_file", t=track, path=path)
        same = [x for x in items if clip_path(x) == path]
        other = len(items) - len(same)
        if other:
            log(tr("other_files", t=track, n=other))
        return it, path, same

    def destination(combo_index, src_track, src_path):
        if combo_index == 0:
            return src_track, True
        if combo_index == 1:
            import soundfile as sf
            tl.AddTrack("audio", "mono" if sf.info(src_path).channels == 1 else "stereo")
            t = tl.GetTrackCount("audio")
            tl.SetTrackName("audio", t, f"{tl.GetTrackName('audio', src_track)} debleed")
            return t, False
        return combo_index - 1, False

    inclusive_end = {"v": None}  # как Resolve понимает endFrame, выясняется на первом куске

    dbg = {"on": False, "events": []}

    def describe(x):
        if not x:
            return None
        out = {}
        for name in ("GetStart", "GetDuration", "GetLeftOffset", "GetName"):
            try:
                out[name] = getattr(x, name)()
            except Exception as e:
                out[name] = f"ERR {e}"
        return out

    def trace(event, **kw):
        if dbg["on"]:
            dbg["events"].append(dict(event=event, **kw))

    def track_state(track):
        items = tl.GetItemListInTrack("audio", track) or []
        return {"count": len(items), "first": [describe(x) for x in items[:3]]}

    def append(mp, infos):
        raw = mp.AppendToTimeline(infos) or []
        trace("append", asked=[{k: v for k, v in i.items() if k != "mediaPoolItem"} for i in infos[:5]],
              asked_count=len(infos), got_count=len(raw), got=[describe(x) for x in raw[:5]])
        return [x for x in raw if x]

    def num(v, what):
        if v is None:
            raise UserError("no_value", what=tr(what))
        return int(v)

    def snapshot(it):
        """Всё, что нужно о клипе, читаем до того, как его трогать: у удалённого клипа Resolve этого не отдаёт."""
        return {"mpi": it.GetMediaPoolItem(), "left": num(it.GetLeftOffset(), "what_left"),
                "start": num(it.GetStart(), "what_start"), "dur": num(it.GetDuration(), "what_dur")}

    def info_for(snap, track, off, ln, inc):
        return {"mediaPoolItem": snap["mpi"], "startFrame": int(snap["left"] + off),
                "endFrame": int(snap["left"] + off + ln - (1 if inc else 0)),
                "trackIndex": int(track), "recordFrame": int(snap["start"] + off), "mediaType": 2}

    def restore(mp, snap, track):
        """Вернуть клип целиком, если нарезка сорвалась."""
        inc = bool(inclusive_end["v"])
        if append(mp, [info_for(snap, track, 0, snap["dur"], inc)]):
            log(tr("restored", t=track))
        else:
            log(tr("restore_failed", t=track))

    def lay_pieces(mp, snap, pieces, track, colors):
        trace("lay_pieces", track=track, snap={k: v for k, v in snap.items() if k != "mpi"},
              n_pieces=len(pieces), first_pieces=pieces[:5], inclusive=inclusive_end["v"], track_now=track_state(track))
        placed = []
        rest = pieces
        if inclusive_end["v"] is None:
            off, ln = pieces[0][0], pieces[0][1]
            first = append(mp, [info_for(snap, track, off, ln, False)])
            if not first:
                raise UserError("not_accepted", t=track)
            if num(first[0].GetDuration(), "what_piece_dur") != ln:
                tl.DeleteClips(first, False)
                first = append(mp, [info_for(snap, track, off, ln, True)])
                inclusive_end["v"] = True
            else:
                inclusive_end["v"] = False
            if not first or num(first[0].GetStart(), "what_piece_start") != snap["start"] + off \
                    or num(first[0].GetDuration(), "what_piece_dur") != ln:
                if first:
                    tl.DeleteClips(first, False)
                inclusive_end["v"] = None
                raise UserError("misplaced")
            placed += first
            rest = pieces[1:]
        for c in range(0, len(rest), 200):
            chunk = rest[c:c + 200]
            got = append(mp, [info_for(snap, track, o, l, inclusive_end["v"]) for o, l, _, _ in chunk])
            bad = len(got) != len(chunk) or any(
                num(g.GetStart(), "what_piece_start") != snap["start"] + o for g, (o, _, _, _) in zip(got, chunk))
            if bad:
                tl.DeleteClips(placed + got, False)
                raise UserError("misplaced_batch", t=track, b=c // 200 + 1)
            placed += got
        n_off = 0
        for item, (_, _, on, kind) in zip(placed, pieces):
            if not on:
                item.SetClipEnabled(False)
                n_off += 1
            if colors and kind in PIECE_COLORS:
                item.SetClipColor(PIECE_COLORS[kind])
        return len(placed), n_off

    def remove_old_markers():
        n = 0
        try:
            for frame, m in list((tl.GetMarkers() or {}).items()):
                if m.get("note") == MARKER_NOTE or m.get("name") in ("Говорят двое", "Говорят трое", "Говорят оба"):
                    if tl.DeleteMarkerAtFrame(frame):
                        n += 1
        except Exception as e:
            log(tr("markers_failed", e=e))
        return n

    def merge_pieces(track, path, mp):
        """Склеить обратно куски прошлой нарезки: подряд идущие клипы одного файла без сдвига."""
        items = [x for x in (tl.GetItemListInTrack("audio", track) or []) if clip_path(x) == path]
        items.sort(key=lambda x: x.GetStart())
        groups, cur = [], []
        for x in items:
            if cur:
                prev = cur[-1]
                if x.GetStart() == prev.GetStart() + prev.GetDuration() and \
                        x.GetLeftOffset() - x.GetStart() == prev.GetLeftOffset() - prev.GetStart():
                    cur.append(x)
                    continue
                groups.append(cur)
            cur = [x]
        if cur:
            groups.append(cur)
        merged = 0
        for g in groups:
            if len(g) < 2:
                continue
            first, last = snapshot(g[0]), snapshot(g[-1])
            whole = dict(first, dur=last["start"] + last["dur"] - first["start"])
            if not tl.DeleteClips(g, False):
                raise UserError("merge_denied", t=track)
            if not append(mp, [info_for(whole, track, 0, whole["dur"], bool(inclusive_end["v"]))]):
                raise UserError("merge_failed", t=track)
            merged += len(g)
        if merged:
            log(tr("merged", t=track, n=merged))

    mode = {"analyze_only": False}

    def on_analyze(ev):
        mode["analyze_only"] = True
        try:
            on_run(ev)
        finally:
            mode["analyze_only"] = False

    def on_run(ev):
        dbg["on"] = bool(itm["Debug"].Checked)
        dbg["events"] = []
        try:
            run_inner(ev)
        finally:
            if dbg["on"]:
                import json
                path = os.path.join(os.path.dirname(SETTINGS_PATH), "debleed_run_debug.json")
                try:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "w", encoding="utf-8") as fh:
                        json.dump({"log": log_lines, "events": dbg["events"]}, fh, ensure_ascii=False,
                                  indent=1, default=str)
                    log(tr("debug_log", path=path))
                except Exception as e:
                    log(tr("debug_failed", e=e))

    def run_inner(ev):
        log_lines.clear()
        try:
            _np()
            import soundfile  # noqa: F401
        except ImportError as e:
            log(tr("missing_lib", name=e.name))
            return
        if not refresh():
            fill_combos()
            log(tr("open_tl", run=tr("run")))
            return
        if built_for["id"] != tl_id(tl):
            fill_combos()
            log(tr("other_tl", name=tl.GetName(), run=tr("run")))
            return
        if len(tracks) < 2:
            log(tr("less2"))
            return
        removed = remove_old_markers()
        if removed:
            log(tr("markers_removed", n=removed))
        N = n_speakers()
        src_idx = [itm[f"Src{k}"].CurrentIndex + 1 for k in range(1, N + 1)]
        if len(set(src_idx)) != N:
            log(tr("distinct"))
            return
        try:
            info = [main_clip(t) for t in src_idx]  # (самый длинный клип, файл, все клипы из этого файла)
            if len({path for _, path, _ in info}) != N:
                raise UserError("same_file")
            pairs = [(path, (it.GetStart() - it.GetLeftOffset()) / fps) for it, path, _ in info]

            p = default_params()
            p.margin = float(itm["Margin"].Value)
            p.thresh = 24.0 - 2.0 * float(itm["Thresh"].Value)  # ползунок 1..10 → порог 22..4 дБ, 5 → 14 дБ
            p.hold = float(itm["Hold"].Value)
            p.analyze_only = mode["analyze_only"]
            min_off = int(round(itm["MinOff"].Value / 1000 * fps))

            if dbg["on"]:
                p.stats = {}
            masks, quiet, overlaps = analyze_tracks(pairs, p, log, tr)
            if dbg["on"]:
                trace("stats", stats=p.stats)
                for i in range(N):
                    trace("open_share", track=src_idx[i], share=float(masks[i].mean()))
            if getattr(p, "analyze_only", False):
                log(tr("analyze_done"))
                return

            if itm["Backup"].Checked:
                import time
                if tl.DuplicateTimeline(tr("backup_name", name=tl.GetName(), time=time.strftime('%H-%M-%S'))):
                    proj.SetCurrentTimeline(tl)
                    log(tr("backup_ok"))
                else:
                    log(tr("backup_fail"))

            mp = proj.GetMediaPool()
            colors = itm["Colors"].Checked
            for k in range(N):
                idx, (_, path, clips) = src_idx[k], info[k]
                mask, qmask, combo = masks[k], quiet[k], itm[f"Dst{k + 1}"].CurrentIndex
                merge_pieces(idx, path, mp)
                clips = main_clip(idx)[2]
                track, same = destination(combo, idx, path)
                total = off = 0
                snaps = [snapshot(it) for it in clips]
                for it, snap in sorted(zip(clips, snaps), key=lambda x: x[1]["start"]):
                    pieces = clip_pieces(mask, qmask, snap["left"], snap["dur"], fps, min_off)
                    if same:
                        trace("before_delete", track=track, state=track_state(track))
                        deleted = tl.DeleteClips([it], False)
                        trace("after_delete", track=track, result=deleted, state=track_state(track))
                        if not deleted:
                            raise UserError("cut_denied", t=track)
                    try:
                        n_placed, n_disabled = lay_pieces(mp, snap, pieces, track, colors)
                    except Exception:
                        if same:
                            restore(mp, snap, track)
                        raise
                    total += n_placed
                    off += n_disabled
                if not same:
                    try:
                        tl.SetTrackEnable("audio", idx, False)
                    except Exception:
                        pass
                log(tr("track_done", t=track, n=total, off=off))

            log(tr("done"))
            refresh()
            fill_combos()
        except Exception as e:
            import traceback
            trace("exception", tb=traceback.format_exc())
            msg = tr(e.key, **e.kw) if isinstance(e, UserError) else str(e)
            log(tr("error", e=msg))

    def on_close(ev):
        disp.ExitLoop()

    win.On.Mode.CurrentIndexChanged = on_mode
    win.On.Lang.CurrentIndexChanged = on_lang
    win.On.Refresh.Clicked = on_refresh
    win.On.Margin.ValueChanged = on_margin
    win.On.Thresh.ValueChanged = on_thresh
    win.On.Run.Clicked = on_run
    win.On.Analyze.Clicked = on_analyze
    win.On.Close.Clicked = on_close
    win.On.DebleedWin.Close = on_close
    win.Show()
    disp.RunLoop()
    win.Hide()


if "bmd" in globals():
    try:
        resolve  # noqa: F821
    except NameError:
        resolve = bmd.scriptapp("Resolve")  # noqa: F821
    resolve_main(resolve, bmd)  # noqa: F821
