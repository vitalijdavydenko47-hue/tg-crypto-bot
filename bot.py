# language: Python, file: bot.py
# TG crypto bot: TON + TAO/USDT, 1m candles, 15-min forecast, chart image
# Railway-ready: reads BOT_TOKEN from env, binds healthcheck port

import os
import io
import asyncio
import logging
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import aiohttp
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import (
    Message, CallbackQuery,
    InlineKeyboardMarkup, InlineKeyboardButton,
    ReplyKeyboardMarkup, KeyboardButton,
    BotCommand, MenuButtonCommands,
    BufferedInputFile,
)

# ─────────────────────────────────────────────────────────────
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"

SYMBOLS = {"TON": "TONUSDT", "TAO": "TAOUSDT"}
INTERVAL = "1m"
CANDLE_LIMIT = 300
FORECAST_HORIZON_MIN = 15

WEIGHTS = {
    "rsi": 0.18, "ema_cross": 0.22, "macd": 0.20,
    "bollinger": 0.15, "momentum": 0.15, "volume": 0.10,
}
MAX_CONFIDENCE = 0.92

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("crypto_bot")
dp = Dispatcher()


# ─────────────────────────────────────────────────────────────
async def fetch_ohlcv(symbol: str, interval: str = INTERVAL,
                      limit: int = CANDLE_LIMIT) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=20)) as s:
        async with s.get(BINANCE_KLINES, params=params) as r:
            r.raise_for_status()
            raw = await r.json()

    rows = [{
        "ts": int(k[0]), "open": float(k[1]), "high": float(k[2]),
        "low": float(k[3]), "close": float(k[4]), "volume": float(k[5]),
    } for k in raw]

    df = (pd.DataFrame(rows).drop_duplicates(subset="ts")
          .sort_values("ts").reset_index(drop=True))
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    al = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def ema(close, span):
    return close.ewm(span=span, adjust=False).mean()


def macd(close, fast=12, slow=26, signal=9):
    m = ema(close, fast) - ema(close, slow)
    s = m.ewm(span=signal, adjust=False).mean()
    return m, s, m - s


def bollinger(close, period=20, mult=2.0):
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    up, lo = ma + mult*sd, ma - mult*sd
    return up, ma, lo, (up - lo) / ma


def atr(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def add_indicators(df):
    c = df["close"]
    df["rsi"] = rsi(c, 14)
    df["ema9"] = ema(c, 9)
    df["ema21"] = ema(c, 21)
    df["ema50"] = ema(c, 50)
    df["macd"], df["macd_sig"], df["macd_hist"] = macd(c)
    df["bb_up"], df["bb_ma"], df["bb_lo"], df["bb_w"] = bollinger(c)
    df["atr"] = atr(df, 14)
    df["ret_6"] = c.pct_change(6)
    df["vol_ma"] = df["volume"].rolling(20).mean()
    return df


def _swing_levels(df, lookback=60):
    w = df.tail(lookback)
    highs, lows = w["high"].values, w["low"].values
    ph, pl = [], []
    for i in range(2, len(w) - 2):
        if highs[i] == max(highs[i-2:i+3]):
            ph.append(highs[i])
        if lows[i] == min(lows[i-2:i+3]):
            pl.append(lows[i])
    return {
        "resistance": sorted(ph)[-3:] if ph else [w["high"].max()],
        "support": sorted(pl)[:3] if pl else [w["low"].min()],
    }


def forecast(df):
    df = add_indicators(df.copy())
    last, prev = df.iloc[-1], df.iloc[-2]
    votes = {}

    rv = last["rsi"]
    votes["rsi"] = 1 if rv <= 30 else (-1 if rv >= 70 else 0)

    bias = 1 if last["ema9"] > last["ema21"] else -1
    slope_ok = (last["ema21"] > prev["ema21"]) if bias > 0 \
               else (last["ema21"] < prev["ema21"])
    votes["ema_cross"] = bias if slope_ok else 0

    h, hp = last["macd_hist"], prev["macd_hist"]
    votes["macd"] = 1 if (h > 0 and h > hp) else (-1 if (h < 0 and h < hp) else 0)

    price = last["close"]
    rng = last["bb_up"] - last["bb_lo"]
    pos = (price - last["bb_lo"]) / rng if rng else 0.5
    if pos <= 0.1 and last["bb_w"] < 0.15:
        votes["bollinger"] = 1
    elif pos >= 0.9:
        votes["bollinger"] = -1
    else:
        votes["bollinger"] = 0

    mom = last["ret_6"]
    votes["momentum"] = 1 if mom > 0.001 else (-1 if mom < -0.001 else 0)

    vr = last["volume"] / last["vol_ma"] if last["vol_ma"] else 1.0
    votes["volume"] = 1 if vr > 1.5 else 0

    score = sum(WEIGHTS[k] * votes[k] for k in votes)
    if votes["volume"] == 1:
        score *= 1.15

    direction = "LONG" if score > 0.05 else ("SHORT" if score < -0.05 else "NEUTRAL")
    conf = round(min(abs(score) / sum(WEIGHTS.values()), MAX_CONFIDENCE) * 100, 1)

    atr_val = float(last["atr"])
    move = 1.5 * atr_val * np.sqrt(FORECAST_HORIZON_MIN)

    return {
        "direction": direction, "confidence": conf, "score": round(score, 4),
        "price": float(price), "rsi": round(float(rv), 1),
        "macd_hist": round(float(h), 6), "bb_pos": round(float(pos), 3),
        "momentum_6m": round(float(mom)*100, 3), "vol_ratio": round(float(vr), 2),
        "votes": votes,
        "support": [float(x) for x in _swing_levels(df)["support"]],
        "resistance": [float(x) for x in _swing_levels(df)["resistance"]],
        "stop_long": float(price - 0.75*move), "target_long": float(price + 1.5*move),
        "stop_short": float(price + 0.75*move), "target_short": float(price - 1.5*move),
        "atr": round(atr_val, 6), "candles_used": len(df),
    }


def render_chart(asset, df, f):
    df = add_indicators(df.copy()).tail(120).reset_index(drop=True)
    x = np.arange(len(df))

    fig, (ax, ax_v) = plt.subplots(
        2, 1, figsize=(10, 6.8), dpi=130,
        gridspec_kw={"height_ratios": [3.2, 1]}, sharex=True)
    fig.patch.set_facecolor("#0d1117")
    for a in (ax, ax_v):
        a.set_facecolor("#0d1117")
        a.tick_params(colors="#c9d1d9", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#30363d")

    for i, row in df.iterrows():
        color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
        ax.plot([i, i], [row["low"], row["high"]], color=color, lw=0.7)
        ax.add_patch(Rectangle(
            (i - 0.3, min(row["open"], row["close"])),
            0.6, max(abs(row["close"] - row["open"]), 1e-9),
            color=color, edgecolor=color))

    ax.plot(x, df["ema9"], color="#ffd54f", lw=1.0)
    ax.plot(x, df["ema21"], color="#4fc3f7", lw=1.0)
    ax.plot(x, df["ema50"], color="#ba68c8", lw=1.0)
    ax.plot(x, df["bb_up"], color="#546e7a", lw=0.7, ls="--")
    ax.plot(x, df["bb_lo"], color="#546e7a", lw=0.7, ls="--")

    price = f["price"]
    ax.axhline(price, color="#ffffff", lw=0.8, ls=":", alpha=0.6)
    proj_x_end = len(df) - 1 + FORECAST_HORIZON_MIN

    if f["direction"] == "LONG":
        ax.plot([len(df)-1, proj_x_end], [price, f["target_long"]],
                color="#26a69a", lw=1.6)
        ax.axhline(f["stop_long"], color="#ef5350", lw=0.9, ls="--")
        ax.axhline(f["target_long"], color="#26a69a", lw=0.9, ls="--")
    elif f["direction"] == "SHORT":
        ax.plot([len(df)-1, proj_x_end], [price, f["target_short"]],
                color="#ef5350", lw=1.6)
        ax.axhline(f["stop_short"], color="#ef5350", lw=0.9, ls="--")
        ax.axhline(f["target_short"], color="#26a69a", lw=0.9, ls="--")

    for s in f["support"]:
        ax.axhline(s, color="#26a69a", lw=0.5, alpha=0.35)
    for r in f["resistance"]:
        ax.axhline(r, color="#ef5350", lw=0.5, alpha=0.35)

    arrow = {"LONG": "🟢", "SHORT": "🔴", "NEUTRAL": "⚪"}[f["direction"]]
    ax.set_title(f"{asset}/USDT  {arrow} {f['direction']}  conf {f['confidence']}%  "
                 f"price {price:.6g}  h={FORECAST_HORIZON_MIN}m",
                 color="#e6edf3", fontsize=11, loc="left", pad=8)

    vol_colors = ["#26a69a" if r["close"] >= r["open"] else "#ef5350"
                  for _, r in df.iterrows()]
    ax_v.bar(x, df["volume"], color=vol_colors, width=0.7)

    step = max(1, len(df) // 8)
    ticks = list(range(0, len(df), step))
    ax_v.set_xticks(ticks)
    ax_v.set_xticklabels([df["dt"].iloc[i].strftime("%H:%M") for i in ticks],
                         color="#c9d1d9", fontsize=7)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    fig.text(0.99, 0.01, f"{stamp} · 1m candles · probabilistic, not a guarantee",
             ha="right", va="bottom", fontsize=6.5, color="#6e7681")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def render_text(asset, f):
    arrow = {"LONG": "🟢", "SHORT": "🔴", "NEUTRAL": "⚪"}[f["direction"]]
    s = lambda v: f"{v:.6g}"
    if f["direction"] == "LONG":
        plan = f"Стоп: {s(f['stop_long'])}   Тейк: {s(f['target_long'])}"
    elif f["direction"] == "SHORT":
        plan = f"Стоп: {s(f['stop_short'])}   Тейк: {s(f['target_short'])}"
    else:
        plan = "Сигнала нет — рынок в балансе."

    v = f["votes"]
    v_line = " ".join(f"{k[0].upper()}:{'+' if v[k]>0 else ('-' if v[k]<0 else '0')}"
                      for k in v)
    sup = " · ".join(s(x) for x in f["support"])
    res = " · ".join(s(x) for x in f["resistance"])

    return (f"{arrow} <b>{asset}/USDT — {f['direction']}</b>  на {FORECAST_HORIZON_MIN} мин\n"
            f"Уверенность: <b>{f['confidence']}%</b>   score {f['score']:+.3f}\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"Цена: <code>{s(f['price'])}</code>\n"
            f"RSI {f['rsi']} · MACD hist {f['macd_hist']:+.6f}\n"
            f"BB pos {f['bb_pos']} · Момент 6м {f['momentum_6m']:+.3f}%\n"
            f"Объём×20MA {f['vol_ratio']} · ATR {s(f['atr'])}\n"
            f"Голоса: <code>{v_line}</code>\n"
            f"━━━━━━━━━━━━━━━━━\n"
            f"Поддержка: {sup}\n"
            f"Сопротивление: {res}\n"
            f"{plan}\n"
            f"<i>Горизонт 15 мин, 1m-свечи. Вероятностный скор, не гарантия.</i>")


def reply_kb():
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="📈 Прогноз TON")],
                  [KeyboardButton(text="📈 Прогноз TAO")]],
        resize_keyboard=True,
        input_field_placeholder="Выбери прогноз…",
    )


async def setup_menu(bot):
    await bot.set_my_commands([
        BotCommand(command="start", description="Открыть меню прогнозов"),
        BotCommand(command="ton", description="Прогноз TON на 15 мин"),
        BotCommand(command="tao", description="Прогноз TAO на 15 мин"),
    ])
    await bot.set_chat_menu_button(menu_button=MenuButtonCommands())


async def run_forecast(bot, chat_id, asset):
    try:
        await bot.send_message(chat_id, f"⏳ {asset}: тяну 1m-свечи и считаю…")
        df = await fetch_ohlcv(SYMBOLS[asset])
        if df.empty or len(df) < 60:
            await bot.send_message(chat_id, f"⚠️ {asset}: мало данных.")
            return
        f = forecast(df)
        png = render_chart(asset, df, f)
        txt = render_text(asset, f)
        await bot.send_photo(chat_id,
            photo=BufferedInputFile(png, filename=f"{asset}.png"),
            caption=txt, parse_mode="HTML")
    except Exception as e:
        log.exception("forecast failed %s", asset)
        await bot.send_message(chat_id, f"⚠️ {asset}: ошибка — {e}")


@dp.message(Command("start"))
async def cmd_start(msg: Message):
    await msg.answer(
        "Бот прогноза <b>TON</b> и <b>TAO/USDT</b> на 15 минут.\n\n"
        "Тяну 1m-свечи с Binance, считаю индикаторы, рисую прогноз.\n\n"
        "Жми кнопку внизу 👇", reply_markup=reply_kb())


@dp.message(Command("ton"))
@dp.message(F.text == "📈 Прогноз TON")
async def cmd_ton(msg: Message):
    await run_forecast(msg.bot, msg.chat.id, "TON")


@dp.message(Command("tao"))
@dp.message(F.text == "📈 Прогноз TAO")
async def cmd_tao(msg: Message):
    await run_forecast(msg.bot, msg.chat.id, "TAO")


# ─────────────────────────────────────────────────────────────
# healthcheck HTTP server (Railway requires a bound port)
# ─────────────────────────────────────────────────────────────
class _Health(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def log_message(self, *a): pass


def _start_health_server():
    port = int(os.getenv("PORT", "8080"))
    srv = HTTPServer(("0.0.0.0", port), _Health)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log.info("health server on :%d", port)


async def main():
    if not BOT_TOKEN:
        raise SystemExit("BOT_TOKEN env variable missing")
    _start_health_server()
    bot = Bot(token=BOT_TOKEN)
    await setup_menu(bot)
    log.info("bot starting…")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
