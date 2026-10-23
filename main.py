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

SETUP LIFECYCLE: a setup is REGISTERED when a 2-candle BOS is confirmed by the LTFs
(Model 1 may register early, in the last 8h of candle 2; both models also register from
the last two CLOSED candles). It then stays LIVE through the whole third candle and is
watched for POI touches at ANY time — the evaluation windows only gate when new setups
are first spotted early. Candle 1 is the dealing range (a POI must form inside it) and its
far extreme is the invalidation level: an LTF close beyond it kills the setup. Every alert
carries a code-computed confluence checklist. Active setups are saved to STATE_PATH.
Only ICT/SMC POIs trigger alerts by default (POI_FAMILY=ictsmc); MSNR levels still count
as confluence. POIs must form INSIDE candle 1's range; there is no premium/discount rule.

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
import html
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

DERIV_WS_URL = "wss://api.derivws.com/trading/v1/options/ws/public"  # public data, no app_id/auth needed

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
# How many of the most recent candles to consider when forming POIs (FVG/OB/A-V Level/
# QML/etc). Drastically shortened from the full 300-candle fetch — a 300-candle lookback
# let too many old, still-"unmitigated" POIs accumulate (seen: 35-49 matching POIs for a
# single confirmed bias), which flooded Telegram with 10+ alerts at once. A much shorter
# window keeps only genuinely recent, relevant levels.
POI_LOOKBACK_CANDLES = int(os.environ.get("POI_LOOKBACK_CANDLES", "60"))
# Deriv meters ticks_history separately as "normal" vs "expensive" and the expensive
# bucket is much tighter — a burst of 300-candle requests across 37 symbols right after
# a cold start/restart was enough to exhaust it in under a second (seen in practice).
# Nothing in this bot actually needs more than ~130 candles (BOS_LOOKBACK_CANDLES=100 is
# the deepest lookback used), so the default request size is cut well below the old 300.
CANDLE_FETCH_COUNT = int(os.environ.get("CANDLE_FETCH_COUNT", "150"))
# Minimum seconds between ANY two candle fetches, enforced globally (not per-symbol) —
# this is what actually spreads out a cold-start burst across many symbols.
CANDLE_FETCH_MIN_GAP = float(os.environ.get("CANDLE_FETCH_MIN_GAP", "0.6"))
# Extra pause, shared by every symbol, the moment Deriv reports the expensive-call limit hit.
CANDLE_FETCH_BACKOFF = int(os.environ.get("CANDLE_FETCH_BACKOFF", "15"))


def _choice(env_name: str, default: str, allowed: tuple) -> str:
    value = os.environ.get(env_name, default).strip().lower()
    if value not in allowed:
        logging.getLogger("crt_alert_bot").warning(
            "%s=%r is not one of %s — using default %r", env_name, value, allowed, default)
        return default
    return value


# Candle 1 is the "range" candle of a CRT setup. It is used two ways:
#  - INVALIDATION: "candle1" = a close beyond candle 1's FAR extreme kills the setup
#    (bullish: close below candle 1's low; bearish: close above its high).
#    "candle1_eq" = a close beyond candle 1's 50% instead (tighter).
#    "candle2"    = the old behaviour (candle 2's far extreme).
#  - DEALING RANGE: the setup must FORM inside candle 1's high-low.
#    "inside"  = the POI must sit entirely within candle 1's range (default).
#    "overlap" = the POI only has to overlap candle 1's range.
#    "off"     = no dealing-range filter.
#    There is deliberately NO premium/discount rule.
INVALIDATION_MODE = _choice("INVALIDATION_MODE", "candle1", ("candle1", "candle1_eq", "candle2"))
DEALING_RANGE_MODE = _choice("DEALING_RANGE_MODE", "inside", ("inside", "overlap", "off"))
# Alerts below this checklist score (out of 5) are suppressed. 0 = show the checklist
# but never suppress. Raise it (e.g. 3 or 4) once you've seen typical scores.
MIN_CHECKLIST_SCORE = int(os.environ.get("MIN_CHECKLIST_SCORE", "0"))

# POI families.
ICT_SMC_POIS = {"fvg", "ifvg", "bpr", "order_block", "propulsion_block",
                "breaker_block", "mitigation_block", "rejection_block"}
MSNR_POIS = {"a_level", "v_level", "qml", "qmc"}
# POI_FAMILY: "ictsmc" (default) = only ICT/SMC POIs can TRIGGER an alert. MSNR levels are
# still detected and still count as confluence on the checklist, they just can't fire an
# alert on their own. "all" = MSNR levels can trigger alerts too.
POI_FAMILY = _choice("POI_FAMILY", "ictsmc", ("ictsmc", "all"))

# When several POIs are touched at once, the highest-priority (lowest number) is alerted.
POI_PRIORITY = {
    "breaker_block": 0, "mitigation_block": 1, "order_block": 2, "propulsion_block": 3,
    "ifvg": 4, "bpr": 5, "rejection_block": 6, "fvg": 7,
    "qmc": 8, "qml": 9, "v_level": 10, "a_level": 10,
}
# ICT/SMC types that are NOT the plain FVG (the checklist treats these as "high-grade").
HIGH_GRADE_POIS = {"order_block", "propulsion_block", "breaker_block", "mitigation_block",
                   "ifvg", "bpr", "rejection_block"}

# LTFs used by each model, which LTF's close decides invalidation, and how long a setup
# stays live. A setup is live through the whole THIRD candle (candle 2 start + 2 candles).
MODEL_LTFS = {"daily_crt": ("H4", "H2", "H1"), "4h_crt": ("M30", "M15")}
MODEL_LTFS_REQUIRED = {"daily_crt": 2, "4h_crt": 2}   # how many LTFs must agree
MODEL_REF_TF = {"daily_crt": "H1", "4h_crt": "M15"}   # close of this TF checks invalidation
MODEL_BOS_TF = {"daily_crt": "D1", "4h_crt": "H4"}
MODEL_TITLES = {"daily_crt": "Daily CRT", "4h_crt": "4H CRT"}


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
        self._cache: dict[tuple, tuple] = {}  # (symbol, gran, count, drop_forming) -> (candles, fetched_at)
        self._candle_lock = asyncio.Lock()
        self._last_candle_call = 0.0

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
        await self.ws.send(json.dumps(payload))
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            raw = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
            msg = json.loads(raw)
            if msg.get("req_id") == req_id:
                if msg.get("error"):
                    raise RuntimeError(f"Deriv API error: {msg['error']}")
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

    async def get_candles_cached(self, symbol: str, granularity: int, count: int = CANDLE_FETCH_COUNT,
                                  drop_forming: bool = True, ttl: Optional[float] = None) -> list[Candle]:
        """Cached candle fetch, to stay inside Deriv's 220 requests/minute budget now
        that setups are monitored continuously.
        - Confirmed-candle lists (drop_forming=True) only change when a new candle
          closes, so they are reused until then (no API call).
        - Lists that include the forming candle need a ttl (seconds) to be reused."""
        key = (symbol, granularity, count, drop_forming)
        now = time.time()
        entry = self._cache.get(key)
        if entry:
            candles, fetched_at = entry
            if drop_forming:
                if candles and now < candles[-1].epoch + 2 * granularity + 2:
                    return candles
            elif ttl is not None and now - fetched_at < ttl:
                return candles
        candles = await self.get_candles(symbol, granularity, count, drop_forming)
        self._cache[key] = (candles, now)
        return candles

    async def get_candles(self, symbol: str, granularity: int, count: int = CANDLE_FETCH_COUNT,
                           drop_forming: bool = True) -> list[Candle]:
        # Every candle fetch (this is Deriv's "expensive" ticks_history call) is paced through
        # ONE shared lock, regardless of which symbol or model triggered it. A per-symbol
        # backoff alone does not stop the NEXT symbol's request firing immediately after a
        # RateLimit error — this closes that gap. CANDLE_FETCH_MIN_GAP is generous enough
        # that a cold start (37 symbols x several timeframes, no cache yet) spreads out
        # instead of bursting all at once.
        async with self._candle_lock:
            wait = CANDLE_FETCH_MIN_GAP - (time.monotonic() - self._last_candle_call)
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                resp = await self._send_recv({
                    "ticks_history": symbol,
                    "adjust_start_time": 1,
                    "count": count,
                    "end": "latest",
                    "start": 1,
                    "style": "candles",
                    "granularity": granularity,
                })
            except RuntimeError as exc:
                if "RateLimit" in str(exc):
                    # Back off BEFORE releasing the lock, so every other symbol waiting on
                    # this same lock is slowed down too, not just this one.
                    log.warning("Hit Deriv's ticks_history rate limit — pausing %ds for ALL symbols", CANDLE_FETCH_BACKOFF)
                    await asyncio.sleep(CANDLE_FETCH_BACKOFF)
                self._last_candle_call = time.monotonic()
                raise
            self._last_candle_call = time.monotonic()
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
                        "zone_high": c3.low, "epoch": candles[i].epoch,
                        "armed_epoch": c3.epoch})
        elif c1.low > c3.high:
            out.append({"type": "fvg", "direction": "bearish", "zone_low": c3.high,
                        "zone_high": c1.low, "epoch": candles[i].epoch,
                        "armed_epoch": c3.epoch})
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
                            "zone_high": fvg["zone_high"], "epoch": c.epoch,
                            "armed_epoch": c.epoch})
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
                later = max(bi, si)
                epoch = candles[later].epoch
                armed = candles[min(later + 1, len(candles) - 1)].epoch
                out.append({"type": "bpr", "direction": "bullish", "zone_low": lo, "zone_high": hi,
                            "epoch": epoch, "armed_epoch": armed})
                out.append({"type": "bpr", "direction": "bearish", "zone_low": lo, "zone_high": hi,
                            "epoch": epoch, "armed_epoch": armed})
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
                        "armed_epoch": c.epoch, "swept_extreme": prior_highs[-1]})
        elif c.close < c.open and prior_lows and c.close < prior_lows[-1] and prior_candle.close > prior_candle.open:
            out.append({"type": "order_block", "direction": "bearish", "zone_low": prior_candle.low,
                        "zone_high": prior_candle.high, "epoch": prior_candle.epoch,
                        "armed_epoch": c.epoch, "swept_extreme": prior_lows[-1]})
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
                               "zone_low": ob["zone_low"], "zone_high": ob["zone_high"], "epoch": ob["epoch"],
                               "armed_epoch": c.epoch}
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
                        "epoch": c.epoch,
                        "armed_epoch": candles[min(idx + SWING_N, len(candles) - 1)].epoch})
    for idx, _ in lows:
        c = candles[idx]
        w, b = lower_wick(c), body(c)
        if b > 0 and w >= LONG_WICK_BODY_MULT * b and w >= LONG_WICK_ATR_MULT * a:
            out.append({"type": "rejection_block", "direction": "bullish", "level": c.low + w / 2,
                        "epoch": c.epoch,
                        "armed_epoch": candles[min(idx + SWING_N, len(candles) - 1)].epoch})
    return out


def detect_a_v_levels(candles: list[Candle]) -> list[dict]:
    """MSNR A Level: a peak CLOSE (close[i] > close[i-1] and close[i] >
    close[i+1]) on a close-only line chart — acts as resistance.
    MSNR V Level: the mirror valley close — acts as support.
    Freshness (not yet revisited) is enforced centrally in filter_unmitigated,
    counting only candles AFTER the confirming candle i+1 — the candle right
    after a peak/valley opens at about the same price, so it would otherwise
    always count as a 'revisit'."""
    out = []
    for i in range(1, len(candles) - 1):
        c = candles[i]
        armed = candles[i + 1].epoch
        if c.close > candles[i - 1].close and c.close > candles[i + 1].close:
            out.append({"type": "a_level", "direction": "bearish", "level": c.close,
                        "epoch": c.epoch, "armed_epoch": armed})
        if c.close < candles[i - 1].close and c.close < candles[i + 1].close:
            out.append({"type": "v_level", "direction": "bullish", "level": c.close,
                        "epoch": c.epoch, "armed_epoch": armed})
    return out


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
                            "epoch": candles[idx1].epoch, "confirmed_epoch": candles[j].epoch,
                            "armed_epoch": candles[j].epoch})
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
                            "epoch": candles[idx1].epoch, "confirmed_epoch": candles[j].epoch,
                            "armed_epoch": candles[j].epoch})
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
                out.append({"type": "qmc", "direction": "bearish", "level": qml["level"], "epoch": c.epoch,
                            "armed_epoch": c.epoch})
                break
            if qml["direction"] == "bullish" and c.close > qml["level"]:
                out.append({"type": "qmc", "direction": "bullish", "level": qml["level"], "epoch": c.epoch,
                            "armed_epoch": c.epoch})
                break
    return out


def filter_unmitigated(pois: list[dict], candles: list[Candle]) -> list[dict]:
    """Keep only POIs price has NOT revisited since they finished forming.
    - Zone POIs (FVG/IFVG/BPR/OB/Propulsion/Breaker/Mitigation): mitigated if any
      later candle's range overlaps the zone.
    - Level POIs (A/V Level, QML, QMC, Rejection Block): stale if any later
      candle's range contains the level.
    'Later' means strictly AFTER the POI's armed_epoch (the last candle that
    forms the pattern). Counting from the pattern's own candles was a bug: an
    order block's creating displacement candle always opens inside the block,
    and an FVG's third candle always touches the gap edge, so every zone POI
    was being discarded the moment it formed."""
    idx_by_epoch = {c.epoch: i for i, c in enumerate(candles)}
    out = []
    for poi in pois:
        armed_idx = idx_by_epoch.get(poi.get("armed_epoch", poi["epoch"]))
        if armed_idx is None:
            continue  # pattern's forming candle no longer in the window
        later = candles[armed_idx + 1:]
        if "zone_low" in poi:
            revisited = any(c.low <= poi["zone_high"] and c.high >= poi["zone_low"] for c in later)
        else:
            revisited = any(c.low <= poi["level"] <= c.high for c in later)
        if not revisited:
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


def cap_nearest_per_type(pois: list[dict], current_price: float) -> list[dict]:
    """Keep only the single POI nearest to current price per (type, direction)
    combination. Without this, a confirmed bias could carry 30-50+ technically-
    valid POIs simultaneously, each capable of triggering its own alert —
    exactly what caused a flood of 10+ messages at once. This keeps the most
    actionable (closest-to-price) level of each kind and drops the rest."""
    def distance(poi: dict) -> float:
        if "level" in poi:
            return abs(poi["level"] - current_price)
        mid = (poi["zone_low"] + poi["zone_high"]) / 2
        return abs(mid - current_price)

    nearest: dict[tuple, dict] = {}
    for poi in pois:
        key = (poi["type"], poi["direction"])
        if key not in nearest or distance(poi) < distance(nearest[key]):
            nearest[key] = poi
    return list(nearest.values())


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
# Setup helpers: candle-1 dealing range, invalidation, POI selection, checklist
# ---------------------------------------------------------------------------

_last_log: dict[str, str] = {}


def log_change(key: str, msg: str, *args):
    """Log at INFO only when the message for this key CHANGES — otherwise the
    per-poll 'no BOS' / 'LTFs disagree' lines would repeat every 30 seconds."""
    text = msg % args if args else msg
    if _last_log.get(key) != text:
        _last_log[key] = text
        log.info("%s", text)


def fmt_price(x: float) -> str:
    return ("%.5f" % x).rstrip("0").rstrip(".")


def candle_to_dict(c: Candle) -> dict:
    return {"epoch": c.epoch, "open": c.open, "high": c.high, "low": c.low, "close": c.close}


def compute_invalidation(c1: Candle, c2: Candle, direction: str) -> float:
    """Invalidation level per INVALIDATION_MODE (default: candle 1's FAR extreme)."""
    if INVALIDATION_MODE == "candle2":
        return c2.low if direction == "bullish" else c2.high
    if INVALIDATION_MODE == "candle1_eq":
        return (c1.high + c1.low) / 2
    return c1.low if direction == "bullish" else c1.high


def is_invalidated(close: float, direction: str, level: float) -> bool:
    """A CLOSE beyond the level invalidates the setup (a wick alone does not)."""
    return close < level if direction == "bullish" else close > level


def poi_bounds(poi: dict) -> tuple[float, float]:
    if "zone_low" in poi:
        return poi["zone_low"], poi["zone_high"]
    return poi["level"], poi["level"]


def poi_in_dealing_range(poi: dict, direction: str, c1_low: float, c1_high: float) -> bool:
    """Candle 1's high-low is the dealing range; the setup must FORM inside it.
    See DEALING_RANGE_MODE. (No premium/discount split — by design.)"""
    if DEALING_RANGE_MODE == "off":
        return True
    p_lo, p_hi = poi_bounds(poi)
    if DEALING_RANGE_MODE == "overlap":
        return p_hi >= c1_low and p_lo <= c1_high
    return p_lo >= c1_low and p_hi <= c1_high  # "inside": entirely within candle 1's range


def poi_priority(poi: dict) -> int:
    return POI_PRIORITY.get(poi["type"], 99)


def build_checklist(setup: dict, poi: dict, other_pois: list[dict], flows: dict, m1: Candle) -> list[tuple[str, bool]]:
    """Objective, code-computed confluence checklist. Every item is a yes/no fact
    from the data — nothing subjective, and the AI never writes or alters it."""
    d = setup["direction"]
    c1_low, c1_high = setup["c1"]["low"], setup["c1"]["high"]
    p_lo, p_hi = poi_bounds(poi)
    # Confluence: any OTHER POI of a different type overlapping this one (MSNR levels count),
    # or the same POI confirmed on two timeframes (Model 2 merges M30+M15).
    stacked = "+" in poi["timeframe"] or any(
        o is not poi and o["type"] != poi["type"]
        and poi_bounds(o)[1] >= p_lo and poi_bounds(o)[0] <= p_hi
        for o in other_pois)
    reacted = (m1.close > p_hi) if d == "bullish" else (m1.close < p_lo)
    return [
        (f"All LTFs agree with the {d} bias ({'/'.join(flows.keys())})", all(f == d for f in flows.values())),
        ("POI formed inside candle 1's range", p_lo >= c1_low and p_hi <= c1_high),
        (f"ICT/SMC high-grade POI ({poi['type'].replace('_', ' ')})", poi["type"] in HIGH_GRADE_POIS),
        ("POI confluence (stacked POI or multi-timeframe)", stacked),
        ("M1 closed back out in the bias direction (rejection)", reacted),
    ]


def make_setup(name: str, symbol: str, model: str, direction: str, c1: Candle, c2: Candle,
               flows: dict, c2_len_seconds: int) -> dict:
    return {
        "id": f"{name}|{model}|{c2.epoch}",
        "name": name, "symbol": symbol, "model": model, "direction": direction,
        "c1": candle_to_dict(c1), "c2": candle_to_dict(c2),
        "ltf_flows": flows,
        "invalidation": compute_invalidation(c1, c2, direction),
        "ref_tf": MODEL_REF_TF[model],
        # live through the whole THIRD candle: candle 2 start + 2 candle lengths
        "valid_until": c2.epoch + 2 * c2_len_seconds,
        "created": time.time(),
        "alerted": False,
    }


# ---------------------------------------------------------------------------
# Model setup detection (spec 3.1 / 3.2). These only FIND and CONFIRM setups.
# Watching them for POI touches happens in AlertEngine._monitor_symbol and is NOT
# restricted to the evaluation windows.
# ---------------------------------------------------------------------------

async def _detect_setup(client: DerivClient, name: str, symbol: str, model: str, known_ids: set,
                        in_window: bool) -> Optional[dict]:
    bos_tf = MODEL_BOS_TF[model]
    gran = GRAN[bos_tf]
    label = "Model1" if model == "daily_crt" else "Model2"

    pairs = []
    if in_window:
        # Early read: candle 2 is still forming (spec: evaluate before candle 2 closes).
        forming = await client.get_candles_cached(symbol, gran, count=10, drop_forming=False, ttl=60)
        if len(forming) >= 2:
            pairs.append((forming[-2], forming[-1]))
    # Always also consider the last two CLOSED candles: the setup stays live through
    # the third candle (the current one), so this keeps it alive after the window ends
    # and rebuilds it after a restart.
    closed = await client.get_candles_cached(symbol, gran, count=10, drop_forming=True)
    if len(closed) >= 2:
        pairs.append((closed[-2], closed[-1]))

    now = time.time()
    for c1, c2 in pairs:
        if now >= c2.epoch + 2 * gran:
            continue  # third candle already over
        direction = two_candle_bos(c1, c2)
        setup_id = f"{name}|{model}|{c2.epoch}"
        if direction is None:
            log_change(f"{label}:{name}:{c2.epoch}", "[%s] %s: no 2-candle %s BOS (c1 H/L=%s/%s, c2 H/L=%s/%s)",
                       name, label, bos_tf, fmt_price(c1.high), fmt_price(c1.low),
                       fmt_price(c2.high), fmt_price(c2.low))
            continue
        if setup_id in known_ids:
            continue
        ltfs = MODEL_LTFS[model]
        ltf_candles = {tf: await client.get_candles_cached(symbol, GRAN[tf]) for tf in ltfs}
        flows = {tf: compute_orderflow(c) for tf, c in ltf_candles.items()}
        agree = sum(1 for f in flows.values() if f == direction)
        need = MODEL_LTFS_REQUIRED[model]
        if agree < need:
            log_change(f"{label}:{name}:{c2.epoch}:flows",
                       "[%s] %s: %s BOS=%s but only %d/%d LTFs agree (%s) — not confirmed",
                       name, label, bos_tf, direction, agree, len(ltfs),
                       " ".join(f"{tf}={f}" for tf, f in flows.items()))
            continue
        return make_setup(name, symbol, model, direction, c1, c2, flows, gran)
    return None


async def check_model1(client: DerivClient, name: str, symbol: str, known_ids: set) -> Optional[dict]:
    """Model 1: Daily CRT Continuation (spec 3.1). BOS on D1, LTFs H4/H2/H1 (2 of 3 must agree)."""
    h = datetime.now(timezone.utc).hour
    in_window = MODEL1_WINDOW_UTC_HOURS[0] <= h < MODEL1_WINDOW_UTC_HOURS[1]
    return await _detect_setup(client, name, symbol, "daily_crt", known_ids, in_window)


async def check_model2(client: DerivClient, name: str, symbol: str, known_ids: set) -> Optional[dict]:
    """Model 2: 4H CRT Continuation (spec 3.2). BOS on H4, LTFs M30 AND M15 must both agree."""
    now = datetime.now(timezone.utc)
    minutes_into_h4 = (now.hour % 4) * 60 + now.minute
    in_window = MODEL2_WINDOW_MINUTES_INTO_H4[0] <= minutes_into_h4 < MODEL2_WINDOW_MINUTES_INTO_H4[1]
    return await _detect_setup(client, name, symbol, "4h_crt", known_ids, in_window)


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
# State persistence (dedup keys, cooldowns, AND active setups)
# ---------------------------------------------------------------------------

def load_state() -> dict:
    state: dict = {}
    try:
        with open(STATE_PATH) as f:
            state = json.load(f)
    except Exception:
        pass
    state.setdefault("fired_keys", [])
    state.setdefault("cooldowns", {})
    state.setdefault("active_setups", {})
    return state


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
3. Risk / Invalidation Note (use payload['invalidation'] as the exact level and
   payload['invalidation_rule'] as the rule)
Do NOT restate the checklist or the candle-1 range — they are displayed separately.

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
            return
        except Exception as exc:  # noqa: BLE001
            log.error("Telegram send failed (HTML): %s — retrying as plain text", exc)
        # An alert must never be lost to a formatting problem: retry without HTML.
        try:
            plain = html.unescape(re.sub(r"<[^>]+>", "", text))
            await self.bot.send_message(chat_id=self.chat_id, text=plain)
        except Exception as exc:  # noqa: BLE001
            log.error("Telegram send failed (plain): %s", exc)


def format_alert(setup: dict, payload: dict, ai_text: str, checklist: list[tuple[str, bool]]) -> str:
    poi = payload["poi"]
    level_str = poi.get("level")
    level_str = fmt_price(level_str) if level_str is not None else \
        f'{fmt_price(poi["zone_low"])}–{fmt_price(poi["zone_high"])}'
    d = setup["direction"]
    c1 = setup["c1"]
    score = sum(1 for _, ok in checklist if ok)
    c2_day = datetime.fromtimestamp(setup["c2"]["epoch"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    def esc(s: str) -> str:
        return html.escape(s, quote=False)  # only <, >, & matter inside Telegram HTML text
    lines = [
        f"🎯 <b>{esc(setup['name'])}</b> — {MODEL_TITLES[setup['model']]} ({d.upper()})",
        f"<b>Triggered POI:</b> {poi['type'].replace('_', ' ').title()} ({poi['timeframe']}) @ {level_str}",
        f"<b>Dealing range (candle 1):</b> {fmt_price(c1['low'])} – {fmt_price(c1['high'])}",
        f"<b>Invalidation:</b> close {'below' if d == 'bullish' else 'above'} {fmt_price(setup['invalidation'])}",
        f"<i>Setup candle 2: {c2_day}</i>",
        "",
        f"<b>Checklist {score}/{len(checklist)}</b>",
    ]
    lines += [f"{'✅' if ok else '❌'} {esc(label)}" for label, ok in checklist]
    lines += ["", esc(ai_text)]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

class AlertEngine:
    def __init__(self):
        self.client = DerivClient(DERIV_WS_URL)
        self.notifier = Notifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.state = load_state()
        self._poi_cache: dict[str, tuple] = {}

    async def start(self):
        await self._connect_with_retry()
        await self.client.load_symbol_map(TARGET_ASSETS)
        self._prune_expired()
        restored = len(self.state["active_setups"])
        await self.notifier.send(
            f"✅ CRT alert bot online. Watching {len(self.client.symbol_map)} assets "
            f"(ALERT_MODE={ALERT_MODE}, POI family={POI_FAMILY}, dealing range={DEALING_RANGE_MODE}, "
            f"invalidation={INVALIDATION_MODE}, min checklist={MIN_CHECKLIST_SCORE}). "
            f"Active setups restored: {restored}.")
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
            await self.notifier.send(
                f"💓 CRT alert bot heartbeat — still running. Active setups: {len(self.state['active_setups'])}.")

    async def _poll_loop(self):
        while True:
            try:
                await self._ensure_connected()
                await self.client.keepalive()
                self._prune_expired()
                for name, symbol in list(self.client.symbol_map.items()):
                    await self._scan_symbol(name, symbol)
                    await asyncio.sleep(0.3)  # spread requests within Deriv's rate budget
            except Exception as exc:  # noqa: BLE001
                log.error("Poll loop error: %s", exc)
                await asyncio.sleep(5)
            await asyncio.sleep(POLL_SECONDS)

    def _prune_expired(self):
        now = time.time()
        expired = [sid for sid, s in self.state["active_setups"].items() if now >= s["valid_until"]]
        for sid in expired:
            log.info("Setup expired (third candle over): %s", sid)
            self._drop_setup(sid)
        if expired:
            save_state(self.state)

    def _drop_setup(self, setup_id: str):
        self.state["active_setups"].pop(setup_id, None)
        self._poi_cache.pop(setup_id, None)

    async def _scan_symbol(self, name: str, symbol: str):
        # 1) Find/confirm new setups (cheap: candles are cached until a new candle closes).
        known = set(self.state["active_setups"].keys())
        for check in (check_model1, check_model2):
            try:
                setup = await check(self.client, name, symbol, known)
            except Exception as exc:  # noqa: BLE001
                log.error("%s failed for %s: %s", check.__name__, name, exc)
                if "RateLimit" in str(exc) or "rate" in str(exc).lower():
                    await asyncio.sleep(10)
                continue
            if setup:
                self.state["active_setups"][setup["id"]] = setup
                known.add(setup["id"])
                save_state(self.state)
                log.info("[%s] SETUP REGISTERED %s %s | candle1 %s–%s | invalidation %s | live until %s UTC",
                         name, setup["model"], setup["direction"].upper(),
                         fmt_price(setup["c1"]["low"]), fmt_price(setup["c1"]["high"]),
                         fmt_price(setup["invalidation"]),
                         datetime.fromtimestamp(setup["valid_until"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M"))
        # 2) Watch every live setup for this symbol — at ANY time, not just in a window.
        await self._monitor_symbol(name, symbol)

    def _setup_pois(self, setup: dict, ltf: dict) -> list[dict]:
        """POIs for a live setup: direction-matched, inside candle 1's dealing range,
        capped to the nearest per type. Cached until an LTF candle closes."""
        sig = tuple(c[-1].epoch if c else 0 for c in ltf.values())
        cached = self._poi_cache.get(setup["id"])
        if cached and cached[0] == sig:
            return cached[1]
        direction = setup["direction"]
        c1_low, c1_high = setup["c1"]["low"], setup["c1"]["high"]
        per_tf = {tf: [p for p in gather_pois(candles[-POI_LOOKBACK_CANDLES:], tf)
                       if p["direction"] == direction] for tf, candles in ltf.items()}
        if setup["model"] == "4h_crt":
            pois = merge_confluence(per_tf.get("M30", []), per_tf.get("M15", []))
        else:
            pois = [p for tf in ltf for p in per_tf[tf]]
        total = len(pois)
        pois = [p for p in pois if poi_in_dealing_range(p, direction, c1_low, c1_high)]
        in_range = len(pois)
        ref_close = ltf[setup["ref_tf"]][-1].close
        pois = cap_nearest_per_type(pois, ref_close)
        counts: dict[str, int] = {}
        for p in pois:
            counts[p["type"]] = counts.get(p["type"], 0) + 1
        log_change(f"poi:{setup['id']}",
                   "[%s] %s %s: %d POI(s) in direction, %d inside candle-1 dealing range (%s), kept %d: %s",
                   setup["name"], setup["model"], direction, total, in_range, DEALING_RANGE_MODE,
                   len(pois), counts or "none")
        self._poi_cache[setup["id"]] = (sig, pois)
        return pois

    async def _monitor_symbol(self, name: str, symbol: str):
        setups = [s for s in list(self.state["active_setups"].values()) if s["name"] == name]
        if not setups:
            return
        m1s: Optional[list[Candle]] = None
        for setup in setups:
            model = setup["model"]
            try:
                ltf = {tf: await self.client.get_candles_cached(symbol, GRAN[tf]) for tf in MODEL_LTFS[model]}
            except Exception as exc:  # noqa: BLE001
                log.error("[%s] LTF fetch failed for %s: %s", name, model, exc)
                continue
            if any(not c for c in ltf.values()):
                continue

            # Invalidation: a CLOSE of the reference LTF beyond the level (candle 1's far extreme by default).
            ref_close = ltf[setup["ref_tf"]][-1].close
            if is_invalidated(ref_close, setup["direction"], setup["invalidation"]):
                log.info("[%s] SETUP INVALIDATED %s %s: %s closed at %s, %s the level %s",
                         name, model, setup["direction"], setup["ref_tf"], fmt_price(ref_close),
                         "below" if setup["direction"] == "bullish" else "above",
                         fmt_price(setup["invalidation"]))
                if setup.get("alerted"):
                    await self.notifier.send(
                        f"⚠️ <b>{html.escape(name, quote=False)}</b> — {MODEL_TITLES[model]} "
                        f"{setup['direction']} setup INVALIDATED ({setup['ref_tf']} closed "
                        f"{'below' if setup['direction'] == 'bullish' else 'above'} {fmt_price(setup['invalidation'])}).")
                self._drop_setup(setup["id"])
                save_state(self.state)
                continue

            # LTF orderflow must STILL agree, otherwise pause (it may recover).
            flows = {tf: compute_orderflow(c) for tf, c in ltf.items()}
            agree = sum(1 for f in flows.values() if f == setup["direction"])
            if agree < MODEL_LTFS_REQUIRED[model]:
                log_change(f"flow:{setup['id']}", "[%s] %s paused: only %d/%d LTFs agree now (%s)",
                           name, model, agree, len(flows), " ".join(f"{k}={v}" for k, v in flows.items()))
                continue
            log_change(f"flow:{setup['id']}", "[%s] %s active: LTFs agree (%s)",
                       name, model, " ".join(f"{k}={v}" for k, v in flows.items()))

            pois = self._setup_pois(setup, ltf)
            if not pois:
                continue

            model_key = f"{name}|{model}"
            if time.time() < self.state["cooldowns"].get(model_key, 0):
                continue

            if m1s is None:
                try:
                    m1s = await self.client.get_candles(symbol, GRAN["M1"], count=3)
                except Exception as exc:  # noqa: BLE001
                    log.error("[%s] Failed to fetch M1: %s", name, exc)
                    return
            if not m1s:
                continue

            candidates = []
            for poi in pois:
                if POI_FAMILY == "ictsmc" and poi["type"] not in ICT_SMC_POIS:
                    continue  # MSNR levels still count as confluence, but can't trigger on their own
                key = dedup_key(name, model, poi)
                if key in self.state["fired_keys"]:
                    continue
                for m1 in m1s[-2:]:  # last two closed M1 candles, so a slow loop can't miss a touch
                    if alert_condition_met(m1, poi):
                        others = [p for p in pois if p is not poi]
                        checklist = build_checklist(setup, poi, others, flows, m1)
                        score = sum(1 for _, ok in checklist if ok)
                        if score >= MIN_CHECKLIST_SCORE:
                            candidates.append((poi, m1, key, checklist, score))
                        else:
                            log_change(f"low:{key}", "[%s] %s touch on %s suppressed: checklist %d/%d < MIN_CHECKLIST_SCORE=%d",
                                       name, model, poi["type"], score, len(checklist), MIN_CHECKLIST_SCORE)
                        break
            if not candidates:
                continue
            # Best first: highest-grade POI type, then highest checklist score.
            candidates.sort(key=lambda t: (poi_priority(t[0]), -t[4]))
            poi, m1, key, checklist, score = candidates[0]

            others = sorted((p for p in pois if p is not poi),
                            key=lambda p: abs(sum(poi_bounds(p)) / 2 - ref_close))[:5]
            c1 = setup["c1"]
            payload = {
                "symbol": name, "model": model, "verdict": setup["direction"],
                "candle1_range": {"high": c1["high"], "low": c1["low"],
                                  "equilibrium": (c1["high"] + c1["low"]) / 2},
                "ltf_flows": flows, "poi": poi, "other_pois": others,
                "invalidation": setup["invalidation"],
                "invalidation_rule": f"a {setup['ref_tf']} close "
                                     f"{'below' if setup['direction'] == 'bullish' else 'above'} this level",
            }
            ai_text = await call_ai_narration(payload)
            await self.notifier.send(format_alert(setup, payload, ai_text, checklist))

            setup["alerted"] = True
            self.state["fired_keys"].append(key)
            self.state["fired_keys"] = self.state["fired_keys"][-5000:]  # cap growth
            self.state["cooldowns"][model_key] = time.time() + COOLDOWN_MINUTES * 60
            save_state(self.state)
            log.info("[%s] ALERT SENT %s %s %s checklist %d/%d", name, model, setup["direction"],
                     poi["type"], score, len(checklist))
            # One machine-readable line per alert. Search the logs for ALERT_RECORD to collect
            # them, then compare outcomes by POI type — that is how to find out which concepts
            # actually perform better, rather than assuming.
            log.info("ALERT_RECORD %s", json.dumps({
                "ts": int(time.time()), "symbol": name, "model": model, "direction": setup["direction"],
                "poi_type": poi["type"], "poi_family": "ictsmc" if poi["type"] in ICT_SMC_POIS else "msnr",
                "poi_tf": poi["timeframe"], "poi_low": poi_bounds(poi)[0], "poi_high": poi_bounds(poi)[1],
                "checklist": f"{score}/{len(checklist)}", "c1_low": c1["low"], "c1_high": c1["high"],
                "invalidation": setup["invalidation"], "c2_epoch": setup["c2"]["epoch"],
                "trigger_m1_close": m1.close}))


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
