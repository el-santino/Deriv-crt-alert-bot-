"""
Deriv CRT Continuation Alert Bot
=================================

Sends a Telegram alert the moment price reaches a Point of Interest (POI) that
is aligned with a CONFIRMED CRT continuation bias (Model 1: Daily, Model 2:
4H). The bot never trades — it only watches and notifies.

ALL trading logic (BOS detection, orderflow, POI detection, invalidation
levels) is deterministic and computed in plain Python below. The AI is used
ONLY to word the final Telegram message from a JSON payload the code already
built — it is explicitly forbidden from inventing levels or concepts not in
that payload (see call_ai_narration). If the AI call fails, a plain template
built directly from the payload is sent instead — an alert is never skipped
because of an AI failure.

HONESTY NOTE: the detectors below (BOS, orderflow, FVG/IFVG/BPR, Order/
Propulsion/Breaker/Mitigation/Rejection Blocks, A/V Level, QML) are
deterministic, rule-based implementations of the definitions given for this
build. They are a good-faith, code-exact match to those definitions, not a
certified trading system. QMC ("Quasimodo Confirmation") in particular has no
established public definition anywhere in the ICT/SMC/MSNR material — it is
defined here specifically for this build (see detect_qmc) as the confirming
candle that closes back through an already-identified QML level after price
revisits it. Treat every alert as decision support, not a trade signal. Not
financial advice.
"""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import websockets
from telegram import Bot
from telegram.constants import ParseMode

try:
    from dotenv import load_dotenv
    load_dotenv()  # no-op in production if no .env file is present (e.g. on Railway)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("crt_alert_bot")


def _clean(value: str) -> str:
    """Strip ALL whitespace anywhere in the string (not just leading/trailing) —
    a value can pick up an embedded newline mid-string from a wrapped copy-paste."""
    return re.sub(r"\s+", "", value)


TELEGRAM_BOT_TOKEN = _clean(os.environ["TELEGRAM_BOT_TOKEN"])
TELEGRAM_CHAT_ID = _clean(os.environ["TELEGRAM_CHAT_ID"])
AI_API_KEY = _clean(os.environ["AI_API_KEY"])
AI_PROVIDER = os.environ.get("AI_PROVIDER", "openai").strip().lower()

if not re.fullmatch(r"\d+:[A-Za-z0-9_-]+", TELEGRAM_BOT_TOKEN):
    bad = sorted(set(ch for ch in TELEGRAM_BOT_TOKEN if not re.match(r"[A-Za-z0-9_:-]", ch)))
    bad_points = [(ch, hex(ord(ch))) for ch in bad]
    raise ValueError(
        f"TELEGRAM_BOT_TOKEN doesn't look like a real Telegram token. "
        f"Unexpected character(s): {bad_points!r}. Re-copy it fresh from @BotFather — "
        f"real tokens only contain digits, letters, underscores, hyphens, and one colon."
    )

DERIV_WS_URL = "wss://api.derivws.com/trading/v1/options/ws/public"  # Deriv current public market-data WebSocket

ALERT_MODE = os.environ.get("ALERT_MODE", "touch").strip().lower()  # "touch" or "reaction"
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "30"))
COOLDOWN_MINUTES = int(os.environ.get("COOLDOWN_MINUTES", "60"))
STATE_PATH = os.environ.get("STATE_PATH", "./state.json").strip()
# Railway's filesystem is ephemeral — state (dedup keys, cooldowns) only survives
# a redeploy if STATE_PATH points at a mounted Volume. Without one, the bot just
# starts with a clean slate on every deploy, which is safe (worst case: a POI
# that already fired once could fire again after a redeploy).

_DEFAULT_SYMBOLS = [
    "Volatility 10 Index", "Volatility 25 Index", "Volatility 50 Index",
    "Volatility 75 Index", "Volatility 100 Index",
    "Volatility 10 (1s) Index", "Volatility 15 (1s) Index", "Volatility 25 (1s) Index",
    "Volatility 30 (1s) Index", "Volatility 50 (1s) Index", "Volatility 75 (1s) Index",
    "Volatility 90 (1s) Index", "Volatility 100 (1s) Index",
    "Boom 50 Index", "Boom 150 Index", "Boom 300 Index", "Boom 500 Index",
    "Boom 600 Index", "Boom 900 Index", "Boom 1000 Index",
    "Crash 50 Index", "Crash 150 Index", "Crash 300 Index", "Crash 500 Index",
    "Crash 600 Index", "Crash 900 Index", "Crash 1000 Index",
    "Step Index 100", "Step Index 200", "Step Index 300", "Step Index 400", "Step Index 500",
    "Jump 10 Index", "Jump 25 Index", "Jump 50 Index", "Jump 75 Index", "Jump 100 Index",
]
_symbols_env = os.environ.get("SYMBOLS", "").strip()
if _symbols_env:
    TARGET_ASSETS = [_clean(name) if False else name.strip() for name in _symbols_env.split(",") if name.strip()]
    # (sanitize whitespace-only artifacts per name, but keep internal spaces like "Volatility 75 Index")
    TARGET_ASSETS = [re.sub(r"\s+", " ", name).strip() for name in TARGET_ASSETS]
else:
    TARGET_ASSETS = _DEFAULT_SYMBOLS

GRAN = {"M1": 60, "M15": 900, "M30": 1800, "H1": 3600, "H2": 7200, "H4": 14400, "D1": 86400}

SWING_N = 3            # fractal: a swing point is the extreme of 2*N+1 confirmed candles
ATR_PERIOD = 14
DISPLACEMENT_ATR_MULT = 1.5
LONG_WICK_BODY_MULT = 2
LONG_WICK_ATR_MULT = 1
MODEL1_WINDOW_UTC_HOURS = (16, 24)   # evaluate Model 1 setup only in this UTC hour range
MODEL2_WINDOW_MINUTES_INTO_H4 = (180, 240)  # last 60 min of the 4H candle
BOS_LOOKBACK_CANDLES = 100  # how far back to search for the most recent structure break


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Candle:
    epoch: int
    open: float
    high: float
    low: float
    close: float


# ---------------------------------------------------------------------------
# Deriv client
# ---------------------------------------------------------------------------

class DerivClient:
    def __init__(self, url: str):
        self.url = url
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.symbol_map: dict[str, str] = {}
        self._req_id = 0

    async def connect(self):
        # ping_interval=None disables the low-level WebSocket protocol ping/pong.
        # Deriv's current public endpoint does not appear to answer that protocol-level
        # ping, which was causing a disconnect roughly every 30s (20s wait + 10s
        # timeout) regardless of actual connection health. Deriv's own documented
        # keepalive is the application-level {"ping": 1} API call instead — see
        # DerivClient.keepalive(), called once per poll cycle in AlertEngine._poll_loop.
        self.ws = await websockets.connect(self.url, ping_interval=None)
        log.info("Connected to Deriv WebSocket")

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    async def _send_recv(self, payload: dict, timeout: float = 20.0) -> dict:
        req_id = self._next_id()
        payload = {**payload, "req_id": req_id}

        # Diagnostic logging: if Deriv reports a validation error, this records
        # the EXACT request that this running process sent. This is important
        # because the current get_candles() request does NOT send subscribe=1.
        log.info("Deriv request req_id=%s: %s", req_id, payload)
        await self.ws.send(json.dumps(payload))

        start = time.monotonic()
        while time.monotonic() - start < timeout:
            raw = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
            msg = json.loads(raw)
            if msg.get("req_id") == req_id:
                if msg.get("error"):
                    log.error(
                        "Deriv API error for req_id=%s. Sent=%s | Response=%s | echo_req=%s",
                        req_id,
                        payload,
                        msg.get("error"),
                        msg.get("echo_req"),
                    )
                    raise RuntimeError(
                        f"Deriv API error: {msg['error']} | request={payload} | "
                        f"echo_req={msg.get('echo_req')}"
                    )
                return msg
        raise TimeoutError(f"Deriv API request timed out: {payload}")

    async def load_symbol_map(self, display_names: list[str]):
        resp = await self._send_recv({"active_symbols": "brief"})
        available = resp.get("active_symbols", [])
        log.info("Deriv active_symbols returned %d total symbols", len(available))
        by_display = {s["underlying_symbol_name"]: s["underlying_symbol"] for s in available}
        normalized = {s["underlying_symbol_name"].lower().replace(" ", ""): s["underlying_symbol"]
                      for s in available}
        for name in display_names:
            if name in by_display:
                self.symbol_map[name] = by_display[name]
                continue
            key = name.lower().replace(" ", "")
            if key in normalized:
                self.symbol_map[name] = normalized[key]
                continue
            log.warning("Could not resolve symbol for '%s' — skipping.", name)
        log.info("Resolved %d/%d target assets to live symbols", len(self.symbol_map), len(display_names))

    async def keepalive(self):
        """Deriv's documented application-level keepalive — a plain {"ping": 1}
        request. Called once per poll cycle instead of relying on the low-level
        WebSocket protocol ping, which this endpoint does not appear to answer."""
        await self._send_recv({"ping": 1}, timeout=10)

    async def get_candles(self, symbol: str, granularity: int, count: int = 300,
                           drop_forming: bool = True) -> list[Candle]:
        resp = await self._send_recv({
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": "latest",
            "start": 1,
            "style": "candles",
            "granularity": granularity,
            # New Deriv API: explicitly request a one-time historical response.
            # This avoids the subscribe validation issue seen on the legacy endpoint.
            "subscribe": 0,
        })
        raw = resp.get("candles", [])
        candles = [Candle(epoch=c["epoch"], open=float(c["open"]), high=float(c["high"]),
                           low=float(c["low"]), close=float(c["close"])) for c in raw]
        if drop_forming and candles and (candles[-1].epoch + granularity) > time.time():
            candles = candles[:-1]
        return candles


# ---------------------------------------------------------------------------
# Core primitives: swing points, ATR, wicks, BOS, orderflow
# ---------------------------------------------------------------------------

def find_swing_points(candles: list[Candle], n: int = SWING_N):
    """A swing high/low is the extreme of 2*N+1 confirmed candles (N on each side)."""
    highs, lows = [], []
    for i in range(n, len(candles) - n):
        window = candles[i - n:i + n + 1]
        if candles[i].high == max(c.high for c in window):
            highs.append((i, candles[i].high))
        if candles[i].low == min(c.low for c in window):
            lows.append((i, candles[i].low))
    return highs, lows


def atr(candles: list[Candle], period: int = ATR_PERIOD) -> Optional[float]:
    if len(candles) < period + 1:
        return None
    trs = []
    for i in range(1, len(candles)):
        c, prev = candles[i], candles[i - 1]
        trs.append(max(c.high - c.low, abs(c.high - prev.close), abs(c.low - prev.close)))
    recent = trs[-period:]
    return sum(recent) / len(recent) if recent else None


def body(c: Candle) -> float:
    return abs(c.close - c.open)


def upper_wick(c: Candle) -> float:
    return c.high - max(c.open, c.close)


def lower_wick(c: Candle) -> float:
    return min(c.open, c.close) - c.low


def is_displacement(candle: Candle, candles_up_to_here: list[Candle]) -> bool:
    """DISPLACEMENT = candle body >= 1.5x ATR(14) AND it closes beyond the prior swing point.
    (The 'closes beyond a swing point' half is checked by each caller, which already
    has the relevant swing list in scope; this function checks the ATR half.)"""
    a = atr(candles_up_to_here)
    return a is not None and a > 0 and body(candle) >= DISPLACEMENT_ATR_MULT * a


def two_candle_bos(candle1: Candle, candle2: Candle) -> Optional[str]:
    """2-candle Break of Structure. Wick or sweep is valid; no body close beyond
    candle 1 required. An outside bar (breaks both sides) yields no setup."""
    if candle2.high > candle1.high and candle2.low >= candle1.low:
        return "bullish"
    if candle2.low < candle1.low and candle2.high <= candle1.high:
        return "bearish"
    return None


def compute_orderflow(candles: list[Candle], lookback: int = BOS_LOOKBACK_CANDLES) -> str:
    """3.2a — orderflow rule shared by both models.
    Bullish: the latest confirmed structure break was a close above the nearest
    prior swing high, AND that break's 'protected' opposite swing point has not
    since been closed through. Bearish is the mirror. Minor swings inside a
    pullback do not flip this — only a close through the protected level does,
    which itself registers as a new (opposite) break in this same scan.
    Neutral if no break exists in the lookback window."""
    if len(candles) < SWING_N * 2 + 2:
        return "neutral"
    highs, lows = find_swing_points(candles)
    start = max(0, len(candles) - lookback)
    last_break = None  # (index, direction)
    for i in range(start, len(candles)):
        close = candles[i].close
        prior_highs = [p for idx, p in highs if idx < i]
        prior_lows = [p for idx, p in lows if idx < i]
        if prior_highs and close > prior_highs[-1]:
            last_break = (i, "bullish")
        if prior_lows and close < prior_lows[-1]:
            last_break = (i, "bearish")
    return last_break[1] if last_break else "neutral"


# ---------------------------------------------------------------------------
# POI detectors — 3.3. Each returns dicts with a common shape:
#   {"type", "direction", "epoch", "timeframe", and either
#    ("zone_low","zone_high") for a range, or ("level",) for a single price}
# ---------------------------------------------------------------------------

def detect_fvgs(candles: list[Candle]) -> list[dict]:
    """FVG: 3 consecutive candles where c1.high < c3.low (bullish, gap is
    support) or c1.low > c3.high (bearish, gap is resistance). Zone = the gap."""
    out = []
    for i in range(1, len(candles) - 1):
        c1, c3 = candles[i - 1], candles[i + 1]
        if c1.high < c3.low:
            out.append({"type": "fvg", "direction": "bullish", "zone_low": c1.high,
                        "zone_high": c3.low, "epoch": candles[i].epoch})
        elif c1.low > c3.high:
            out.append({"type": "fvg", "direction": "bearish", "zone_low": c3.high,
                        "zone_high": c1.low, "epoch": candles[i].epoch})
    return out


def detect_ifvgs(candles: list[Candle], fvgs: list[dict]) -> list[dict]:
    """IFVG: an FVG later fully breached by a DISPLACEMENT candle (close beyond
    the far edge of the gap), then it acts as support/resistance in the
    OPPOSITE direction. Zone = the original gap."""
    out = []
    idx_by_epoch = {c.epoch: i for i, c in enumerate(candles)}
    for fvg in fvgs:
        origin = idx_by_epoch.get(fvg["epoch"])
        if origin is None:
            continue
        for j in range(origin + 1, len(candles)):
            c = candles[j]
            breached = (fvg["direction"] == "bullish" and c.close < fvg["zone_low"]) or \
                       (fvg["direction"] == "bearish" and c.close > fvg["zone_high"])
            if breached and is_displacement(c, candles[:j + 1]):
                flipped = "bearish" if fvg["direction"] == "bullish" else "bullish"
                out.append({"type": "ifvg", "direction": flipped, "zone_low": fvg["zone_low"],
                            "zone_high": fvg["zone_high"], "epoch": c.epoch})
                break
    return out


def detect_bprs(candles: list[Candle], fvgs: list[dict]) -> list[dict]:
    """BPR: a bullish FVG and a bearish FVG overlapping, formed within 10
    candles of each other. Zone = the overlap. Can act as support OR
    resistance depending which side price approaches from, so it is emitted
    once per direction."""
    out = []
    idx_by_epoch = {c.epoch: i for i, c in enumerate(candles)}
    bulls = [f for f in fvgs if f["direction"] == "bullish"]
    bears = [f for f in fvgs if f["direction"] == "bearish"]
    for b in bulls:
        bi = idx_by_epoch.get(b["epoch"])
        for s in bears:
            si = idx_by_epoch.get(s["epoch"])
            if bi is None or si is None or abs(bi - si) > 10:
                continue
            lo, hi = max(b["zone_low"], s["zone_low"]), min(b["zone_high"], s["zone_high"])
            if lo < hi:
                epoch = candles[max(bi, si)].epoch
                out.append({"type": "bpr", "direction": "bullish", "zone_low": lo, "zone_high": hi, "epoch": epoch})
                out.append({"type": "bpr", "direction": "bearish", "zone_low": lo, "zone_high": hi, "epoch": epoch})
    return out


def detect_order_blocks(candles: list[Candle]) -> list[dict]:
    """Order Block: the last opposing-color candle before a DISPLACEMENT move
    that breaks the most recent swing point. Zone = that candle's full range
    (high to low). Carries 'swept_extreme' (the swing point it broke) forward
    for Breaker/Mitigation detection."""
    out = []
    highs, lows = find_swing_points(candles)
    for i in range(1, len(candles)):
        c = candles[i]
        if not is_displacement(c, candles[:i + 1]):
            continue
        prior_highs = [p for idx, p in highs if idx < i]
        prior_lows = [p for idx, p in lows if idx < i]
        prior_candle = candles[i - 1]
        if c.close > c.open and prior_highs and c.close > prior_highs[-1] and prior_candle.close < prior_candle.open:
            out.append({"type": "order_block", "direction": "bullish", "zone_low": prior_candle.low,
                        "zone_high": prior_candle.high, "epoch": prior_candle.epoch,
                        "swept_extreme": prior_highs[-1]})
        elif c.close < c.open and prior_lows and c.close < prior_lows[-1] and prior_candle.close > prior_candle.open:
            out.append({"type": "order_block", "direction": "bearish", "zone_low": prior_candle.low,
                        "zone_high": prior_candle.high, "epoch": prior_candle.epoch,
                        "swept_extreme": prior_lows[-1]})
    return out


def detect_propulsion_blocks(obs: list[dict]) -> list[dict]:
    """Propulsion Block: an OB formed INSIDE the range of an earlier OB in the
    same direction — indicates immediate continuation."""
    out = []
    for i, ob in enumerate(obs):
        for earlier in obs[:i]:
            if (earlier["direction"] == ob["direction"] and
                    ob["zone_low"] >= earlier["zone_low"] and ob["zone_high"] <= earlier["zone_high"]):
                out.append({**ob, "type": "propulsion_block"})
                break
    return out


def detect_breaker_and_mitigation_blocks(candles: list[Candle], obs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Breaker Block: an OB whose swing extreme was SWEPT (a wick beyond it,
    close back inside — external liquidity taken), followed by displacement
    breaking structure the OTHER way. The OB flips polarity.
    Mitigation Block: identical, but the structure break happens WITHOUT the
    prior sweep."""
    breakers, mitigations = [], []
    highs, lows = find_swing_points(candles)
    idx_by_epoch = {c.epoch: i for i, c in enumerate(candles)}
    for ob in obs:
        origin = idx_by_epoch.get(ob["epoch"])
        if origin is None or ob.get("swept_extreme") is None:
            continue
        swept = False
        for j in range(origin + 1, len(candles)):
            c = candles[j]
            if ob["direction"] == "bullish" and c.low < ob["zone_low"] and c.close > ob["zone_low"]:
                swept = True
            elif ob["direction"] == "bearish" and c.high > ob["zone_high"] and c.close < ob["zone_high"]:
                swept = True
            if is_displacement(c, candles[:j + 1]):
                prior_highs = [p for idx, p in highs if idx < j]
                prior_lows = [p for idx, p in lows if idx < j]
                opposite_break = (ob["direction"] == "bullish" and prior_lows and c.close < prior_lows[-1]) or \
                                  (ob["direction"] == "bearish" and prior_highs and c.close > prior_highs[-1])
                if opposite_break:
                    flipped = {"type": "breaker_block" if swept else "mitigation_block",
                               "direction": "bearish" if ob["direction"] == "bullish" else "bullish",
                               "zone_low": ob["zone_low"], "zone_high": ob["zone_high"], "epoch": ob["epoch"]}
                    (breakers if swept else mitigations).append(flipped)
                    break
    return breakers, mitigations


def detect_rejection_blocks(candles: list[Candle]) -> list[dict]:
    """Rejection Block: a LONG_WICK candle (wick >= 2x body AND wick >= 1x
    ATR(14)) at a confirmed swing high/low. Alert level = the wick's 50%
    (Consequent Encroachment)."""
    out = []
    highs, lows = find_swing_points(candles)
    a = atr(candles)
    if a is None or a == 0:
        return out
    for idx, _ in highs:
        c = candles[idx]
        w, b = upper_wick(c), body(c)
        if b > 0 and w >= LONG_WICK_BODY_MULT * b and w >= LONG_WICK_ATR_MULT * a:
            out.append({"type": "rejection_block", "direction": "bearish", "level": c.high - w / 2,
                        "epoch": c.epoch})
    for idx, _ in lows:
        c = candles[idx]
        w, b = lower_wick(c), body(c)
        if b > 0 and w >= LONG_WICK_BODY_MULT * b and w >= LONG_WICK_ATR_MULT * a:
            out.append({"type": "rejection_block", "direction": "bullish", "level": c.low + w / 2,
                        "epoch": c.epoch})
    return out


def detect_a_v_levels(candles: list[Candle]) -> list[dict]:
    """MSNR A Level: a peak CLOSE (close[i] > close[i-1] and close[i] >
    close[i+1]) on a close-only line chart — acts as resistance.
    MSNR V Level: the mirror valley close — acts as support.
    Both are 'fresh' only until a later candle's range revisits the level."""
    out = []
    for i in range(1, len(candles) - 1):
        c = candles[i]
        if c.close > candles[i - 1].close and c.close > candles[i + 1].close:
            out.append({"type": "a_level", "direction": "bearish", "level": c.close, "epoch": c.epoch})
        if c.close < candles[i - 1].close and c.close < candles[i + 1].close:
            out.append({"type": "v_level", "direction": "bullish", "level": c.close, "epoch": c.epoch})
    fresh = []
    idx_by_epoch = {c.epoch: i for i, c in enumerate(candles)}
    for lv in out:
        origin = idx_by_epoch.get(lv["epoch"])
        if origin is None:
            continue
        touched = any(c.low <= lv["level"] <= c.high for c in candles[origin + 1:])
        if not touched:
            fresh.append(lv)
    return fresh


def detect_qml(candles: list[Candle]) -> list[dict]:
    """QML (Quasimodo Level, MSNR): bearish = swing high (left shoulder) ->
    higher swing high (head) -> a close below the low between them (breaking
    structure). Level = the left-shoulder high (where liquidity was trapped).
    Bullish is the exact mirror."""
    out = []
    highs, lows = find_swing_points(candles)
    for i in range(len(highs) - 1):
        idx1, shoulder = highs[i]
        idx2, head = highs[i + 1]
        if idx2 <= idx1 or head <= shoulder:
            continue
        between = [p for idx, p in lows if idx1 < idx < idx2]
        if not between:
            continue
        neckline = min(between)
        for j in range(idx2 + 1, len(candles)):
            if candles[j].close < neckline:
                out.append({"type": "qml", "direction": "bearish", "level": shoulder,
                            "epoch": candles[idx1].epoch, "confirmed_epoch": candles[j].epoch})
                break
    for i in range(len(lows) - 1):
        idx1, shoulder = lows[i]
        idx2, head = lows[i + 1]
        if idx2 <= idx1 or head >= shoulder:
            continue
        between = [p for idx, p in highs if idx1 < idx < idx2]
        if not between:
            continue
        neckline = max(between)
        for j in range(idx2 + 1, len(candles)):
            if candles[j].close > neckline:
                out.append({"type": "qml", "direction": "bullish", "level": shoulder,
                            "epoch": candles[idx1].epoch, "confirmed_epoch": candles[j].epoch})
                break
    return out


def detect_qmc(candles: list[Candle], qmls: list[dict]) -> list[dict]:
    """QMC (Quasimodo Confirmation, MSNR) — NO ESTABLISHED PUBLIC DEFINITION
    FOUND; invented for this build. The confirming reaction after price
    returns to an already-identified QML level: the first candle, after
    price revisits that level, which closes back through it in the QML's
    own direction (below it for bearish, above it for bullish). QMC has no
    zone of its own — it's the trigger candle, not a separate price range."""
    out = []
    idx_by_epoch = {c.epoch: i for i, c in enumerate(candles)}
    for qml in qmls:
        conf = idx_by_epoch.get(qml["confirmed_epoch"])
        if conf is None:
            continue
        for j in range(conf + 1, len(candles)):
            c = candles[j]
            if not (c.low <= qml["level"] <= c.high):
                continue
            if qml["direction"] == "bearish" and c.close < qml["level"]:
                out.append({"type": "qmc", "direction": "bearish", "level": qml["level"], "epoch": c.epoch})
                break
            if qml["direction"] == "bullish" and c.close > qml["level"]:
                out.append({"type": "qmc", "direction": "bullish", "level": qml["level"], "epoch": c.epoch})
                break
    return out


def filter_unmitigated(pois: list[dict], candles: list[Candle]) -> list[dict]:
    """Only unmitigated zones count: price must not have traded into a
    zone-based POI (wick or body overlap) since it formed. Level-based POIs
    (A/V Level, QML, Rejection Block, QMC) use their own 'fresh'/one-shot
    logic upstream and pass through here unchanged."""
    idx_by_epoch = {c.epoch: i for i, c in enumerate(candles)}
    out = []
    for poi in pois:
        if "zone_low" not in poi:
            out.append(poi)
            continue
        origin = idx_by_epoch.get(poi["epoch"])
        if origin is None:
            out.append(poi)
            continue
        mitigated = any(c.low <= poi["zone_high"] and c.high >= poi["zone_low"] for c in candles[origin + 1:])
        if not mitigated:
            out.append(poi)
    return out


def gather_pois(candles: list[Candle], timeframe: str) -> list[dict]:
    """Run every detector and return the combined, unmitigated/fresh POI list,
    each tagged with its timeframe."""
    fvgs = detect_fvgs(candles)
    obs = detect_order_blocks(candles)
    breakers, mitigations = detect_breaker_and_mitigation_blocks(candles, obs)
    qmls = detect_qml(candles)
    pois = (
        fvgs
        + detect_ifvgs(candles, fvgs)
        + detect_bprs(candles, fvgs)
        + obs
        + detect_propulsion_blocks(obs)
        + breakers
        + mitigations
        + detect_rejection_blocks(candles)
        + detect_a_v_levels(candles)
        + qmls
        + detect_qmc(candles, qmls)
    )
    pois = filter_unmitigated(pois, candles)
    for p in pois:
        p["timeframe"] = timeframe
    return pois


def merge_confluence(pois_a: list[dict], pois_b: list[dict]) -> list[dict]:
    """If the same POI type+direction overlaps across two timeframes, merge
    into ONE entry listing both timeframes as confluence (Model 2, 3.2.3)."""
    merged = [dict(p) for p in pois_a]
    used_b = set()
    for pa in merged:
        for j, pb in enumerate(pois_b):
            if j in used_b or pb["type"] != pa["type"] or pb["direction"] != pa["direction"]:
                continue
            same_zone = ("zone_low" in pa and "zone_low" in pb and
                         max(pa["zone_low"], pb["zone_low"]) < min(pa["zone_high"], pb["zone_high"]))
            same_level = ("level" in pa and "level" in pb and abs(pa["level"] - pb["level"]) < 1e-9)
            if same_zone or same_level:
                pa["timeframe"] = f"{pa['timeframe']}+{pb['timeframe']}"
                used_b.add(j)
    merged.extend(p for j, p in enumerate(pois_b) if j not in used_b)
    return merged


# ---------------------------------------------------------------------------
# Model checks
# ---------------------------------------------------------------------------

async def check_model1(client: DerivClient, symbol: str) -> Optional[dict]:
    """Model 1: Daily CRT Continuation (spec 3.1)."""
    now = datetime.now(timezone.utc)
    if not (MODEL1_WINDOW_UTC_HOURS[0] <= now.hour < MODEL1_WINDOW_UTC_HOURS[1]):
        return None  # outside the 16:00-24:00 UTC evaluation window — logged once at debug level is noisy, skip
    d1 = await client.get_candles(symbol, GRAN["D1"], count=10, drop_forming=False)
    if len(d1) < 2:
        log.debug("[%s] Model1: not enough D1 candles", symbol)
        return None
    candle1, candle2 = d1[-2], d1[-1]
    direction = two_candle_bos(candle1, candle2)
    if direction is None:
        log.info("[%s] Model1: no 2-candle D1 BOS (c1 H/L=%.5f/%.5f, c2 H/L=%.5f/%.5f)",
                  symbol, candle1.high, candle1.low, candle2.high, candle2.low)
        return None

    h4 = await client.get_candles(symbol, GRAN["H4"])
    h2 = await client.get_candles(symbol, GRAN["H2"])
    h1 = await client.get_candles(symbol, GRAN["H1"])
    flows = {"H4": compute_orderflow(h4), "H2": compute_orderflow(h2), "H1": compute_orderflow(h1)}
    agree = sum(1 for f in flows.values() if f == direction)
    if agree < 2:
        log.info("[%s] Model1: D1 BOS=%s but only %d/3 LTFs agree (H4=%s H2=%s H1=%s) — invalidated",
                  symbol, direction, agree, flows["H4"], flows["H2"], flows["H1"])
        return {"verdict": "invalidated", "model": "daily_crt"}

    pois = [p for p in gather_pois(h4, "H4") + gather_pois(h2, "H2") + gather_pois(h1, "H1")
            if p["direction"] == direction]
    if not pois:
        log.info("[%s] Model1: D1 BOS=%s, LTFs agree (%d/3), but no matching unmitigated POI found",
                  symbol, direction, agree)
    else:
        log.info("[%s] Model1: CONFIRMED %s bias, %d matching POI(s) found", symbol, direction, len(pois))
    invalidation = candle2.low if direction == "bullish" else candle2.high
    return {"verdict": direction, "model": "daily_crt", "symbol": symbol,
            "bos_candle1": candle1, "bos_candle2": candle2, "ltf_flows": flows,
            "pois": pois, "invalidation": invalidation}


async def check_model2(client: DerivClient, symbol: str) -> Optional[dict]:
    """Model 2: 4H CRT Continuation (spec 3.2)."""
    now = datetime.now(timezone.utc)
    minutes_into_h4 = (now.hour % 4) * 60 + now.minute
    if not (MODEL2_WINDOW_MINUTES_INTO_H4[0] <= minutes_into_h4 < MODEL2_WINDOW_MINUTES_INTO_H4[1]):
        return None
    h4 = await client.get_candles(symbol, GRAN["H4"], count=10, drop_forming=False)
    if len(h4) < 2:
        log.debug("[%s] Model2: not enough H4 candles", symbol)
        return None
    candle1, candle2 = h4[-2], h4[-1]
    direction = two_candle_bos(candle1, candle2)
    if direction is None:
        log.info("[%s] Model2: no 2-candle H4 BOS (c1 H/L=%.5f/%.5f, c2 H/L=%.5f/%.5f)",
                  symbol, candle1.high, candle1.low, candle2.high, candle2.low)
        return None

    m30 = await client.get_candles(symbol, GRAN["M30"])
    m15 = await client.get_candles(symbol, GRAN["M15"])
    flow30, flow15 = compute_orderflow(m30), compute_orderflow(m15)
    if not (flow30 == direction and flow15 == direction):
        log.info("[%s] Model2: H4 BOS=%s but M30=%s/M15=%s don't both agree — invalidated",
                  symbol, direction, flow30, flow15)
        return {"verdict": "invalidated", "model": "4h_crt"}

    pois30 = [p for p in gather_pois(m30, "M30") if p["direction"] == direction]
    pois15 = [p for p in gather_pois(m15, "M15") if p["direction"] == direction]
    merged = merge_confluence(pois30, pois15)
    if not merged:
        log.info("[%s] Model2: H4 BOS=%s, M30/M15 agree, but no matching unmitigated POI found",
                  symbol, direction)
    else:
        log.info("[%s] Model2: CONFIRMED %s bias, %d matching POI(s) found", symbol, direction, len(merged))
    invalidation = candle2.low if direction == "bullish" else candle2.high
    return {"verdict": direction, "model": "4h_crt", "symbol": symbol,
            "bos_candle1": candle1, "bos_candle2": candle2,
            "ltf_flows": {"M30": flow30, "M15": flow15},
            "pois": merged, "invalidation": invalidation}


# ---------------------------------------------------------------------------
# Alert triggering (3.4)
# ---------------------------------------------------------------------------

def poi_touched(m1: Candle, poi: dict) -> bool:
    if "zone_low" in poi:
        return m1.low <= poi["zone_high"] and m1.high >= poi["zone_low"]
    return m1.low <= poi["level"] <= m1.high


def alert_condition_met(m1: Candle, poi: dict) -> bool:
    if not poi_touched(m1, poi):
        return False
    if ALERT_MODE == "touch":
        return True
    # "reaction": touch, then that M1 candle closes back out of the zone in the bias direction
    if "zone_low" in poi:
        return m1.close > poi["zone_high"] if poi["direction"] == "bullish" else m1.close < poi["zone_low"]
    return m1.close > poi["level"] if poi["direction"] == "bullish" else m1.close < poi["level"]


def dedup_key(symbol: str, model: str, poi: dict) -> str:
    zone = f'{poi.get("zone_high", poi.get("level"))}|{poi.get("zone_low", poi.get("level"))}'
    return f'{symbol}|{model}|{poi["type"]}|{poi["timeframe"]}|{zone}|{poi["epoch"]}'


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def load_state() -> dict:
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except Exception:
        return {"fired_keys": [], "cooldowns": {}}


def save_state(state: dict):
    try:
        with open(STATE_PATH, "w") as f:
            json.dump(state, f)
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to save state to %s: %s", STATE_PATH, exc)


# ---------------------------------------------------------------------------
# AI narration — words the payload only, never invents levels
# ---------------------------------------------------------------------------

def plain_template(payload: dict) -> str:
    """Fallback if the AI call fails or times out — never skip an alert."""
    poi = payload["poi"]
    poi_level = poi.get("level")
    if poi_level is None:
        poi_level = f'{poi["zone_low"]}-{poi["zone_high"]}'
    lines = [
        f"Bias Verdict: {payload['verdict'].title()}",
        "Key Levels to Watch:",
        f"  Triggered: {poi['type']} ({poi['timeframe']}) @ {poi_level}",
    ]
    for p in payload.get("other_pois", [])[:5]:
        lvl = p.get("level")
        if lvl is None:
            lvl = f'{p["zone_low"]}-{p["zone_high"]}'
        lines.append(f"  Also watch: {p['type']} ({p['timeframe']}) @ {lvl}")
    lines.append(f"Risk/Invalidation Note: invalidated on a close beyond {payload['invalidation']}.")
    return "\n".join(lines)


async def call_ai_narration(payload: dict) -> str:
    prompt = f"""You write Telegram trading alerts. Using ONLY the JSON payload below, write a
message of AT MOST 120 words with exactly these three sections:
1. Bias Verdict (must exactly match payload['verdict'] — do not change it)
2. Key Levels to Watch (ONLY levels/POIs present in the payload — never mention any
   level, POI type, or concept that is not explicitly in this payload)
3. Risk / Invalidation Note (use payload['invalidation'] as the exact level)

PAYLOAD:
{json.dumps(payload, default=str)}
"""
    try:
        if AI_PROVIDER == "gemini":
            text = await asyncio.wait_for(_call_gemini(prompt), timeout=15)
        else:
            text = await asyncio.wait_for(_call_openai(prompt), timeout=15)
        return text
    except Exception as exc:  # noqa: BLE001
        log.error("AI narration failed (%s) — sending plain template instead.", exc)
        return plain_template(payload)


async def _call_openai(prompt: str) -> str:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=AI_API_KEY)
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You write concise, payload-only trading alerts."},
                  {"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


async def _call_gemini(prompt: str) -> str:
    from google import genai
    client = genai.Client(api_key=AI_API_KEY)
    resp = await asyncio.to_thread(client.models.generate_content, model="gemini-3.6-flash", contents=prompt)
    return resp.text.strip()


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

class Notifier:
    def __init__(self, token: str, chat_id: str):
        self.bot = Bot(token=token)
        self.chat_id = chat_id

    async def send(self, text: str):
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=text, parse_mode=ParseMode.HTML)
        except Exception as exc:  # noqa: BLE001
            log.error("Telegram send failed: %s", exc)


def format_alert(symbol: str, payload: dict, ai_text: str) -> str:
    poi = payload["poi"]
    level_str = poi.get("level")
    if level_str is None:
        level_str = f'{poi["zone_low"]}–{poi["zone_high"]}'
    return (
        f"🎯 <b>{symbol}</b> — {payload['model'].replace('_', ' ').title()}\n"
        f"<b>Triggered POI:</b> {poi['type'].replace('_', ' ').title()} ({poi['timeframe']}) @ {level_str}\n\n"
        f"{ai_text}"
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

class AlertEngine:
    def __init__(self):
        self.client = DerivClient(DERIV_WS_URL)
        self.notifier = Notifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.state = load_state()

    async def start(self):
        await self._connect_with_retry()
        await self.client.load_symbol_map(TARGET_ASSETS)
        await self.notifier.send(f"✅ CRT alert bot online. Watching {len(self.client.symbol_map)} assets "
                                  f"(ALERT_MODE={ALERT_MODE}).")
        asyncio.create_task(self._heartbeat_loop())
        await self._poll_loop()

    async def _connect_with_retry(self):
        delay = 2
        while True:
            try:
                await self.client.connect()
                return
            except Exception as exc:  # noqa: BLE001
                log.error("Deriv connect failed (%s). Retrying in %ds", exc, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)

    async def _ensure_connected(self):
        try:
            if self.client.ws is None or self.client.ws.closed:
                raise ConnectionError("socket closed")
        except Exception:
            log.warning("Reconnecting to Deriv...")
            await self._connect_with_retry()

    async def _heartbeat_loop(self):
        while True:
            await asyncio.sleep(24 * 3600)
            await self.notifier.send("💓 CRT alert bot heartbeat — still running.")

    async def _poll_loop(self):
        while True:
            try:
                await self._ensure_connected()
                await self.client.keepalive()
                for name, symbol in list(self.client.symbol_map.items()):
                    await self._scan_symbol(name, symbol)
                    await asyncio.sleep(0.2)  # spread requests within Deriv's rate budget
            except Exception as exc:  # noqa: BLE001
                log.error("Poll loop error: %s", exc)
                await asyncio.sleep(5)
            await asyncio.sleep(POLL_SECONDS)

    async def _scan_symbol(self, name: str, symbol: str):
        for check in (check_model1, check_model2):
            try:
                result = await check(self.client, symbol)
            except Exception as exc:  # noqa: BLE001
                log.error("%s failed for %s: %s", check.__name__, name, exc)
                continue
            if result is None or result["verdict"] == "invalidated" or not result.get("pois"):
                continue

            model_key = f'{name}|{result["model"]}'
            cooldown_until = self.state["cooldowns"].get(model_key, 0)
            if time.time() < cooldown_until:
                continue

            try:
                m1 = await self.client.get_candles(symbol, GRAN["M1"], count=2)
            except Exception as exc:  # noqa: BLE001
                log.error("Failed to fetch M1 for %s: %s", name, exc)
                continue
            if not m1:
                continue
            latest_m1 = m1[-1]

            for poi in result["pois"]:
                if not alert_condition_met(latest_m1, poi):
                    continue
                key = dedup_key(name, result["model"], poi)
                if key in self.state["fired_keys"]:
                    continue

                other = [p for p in result["pois"] if p is not poi][:5]
                payload = {
                    "symbol": name, "model": result["model"], "verdict": result["verdict"],
                    "bos_candle1": result["bos_candle1"], "bos_candle2": result["bos_candle2"],
                    "ltf_flows": result["ltf_flows"], "poi": poi, "other_pois": other,
                    "invalidation": result["invalidation"],
                }
                ai_text = await call_ai_narration(payload)
                await self.notifier.send(format_alert(name, payload, ai_text))

                self.state["fired_keys"].append(key)
                self.state["fired_keys"] = self.state["fired_keys"][-5000:]  # cap growth
                self.state["cooldowns"][model_key] = time.time() + COOLDOWN_MINUTES * 60
                save_state(self.state)
                break  # one alert per model per scan is enough; cooldown handles the rest


async def main():
    engine = AlertEngine()
    while True:
        try:
            await engine.start()
        except Exception as exc:  # noqa: BLE001
            log.exception("Fatal error, restarting in 10s: %s", exc)
            await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(main())
