#!/usr/bin/env python3
"""
Telegram-бот: собирает просмотры оригинальных постов X через SocialData
и присылает минималистичную PNG-карточку с суммой и графиком.

Установка:
    python -m pip install pyTelegramBotAPI requests Pillow matplotlib

Перед запуском задайте переменные окружения:
    export TG_TOKEN="токен Telegram-бота"
    export SOCIALDATA_API_KEY="ключ SocialData"
    python x_impressions_bot.py

Сессии диалога хранятся в памяти процесса и сбрасываются при его перезапуске.
"""

import io
import logging
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import telebot
from PIL import Image, ImageDraw, ImageFont


# Секреты берутся из окружения, а не хранятся в исходном файле.
TG_TOKEN = os.getenv("TG_TOKEN", "").strip()
SOCIALDATA_API_KEY = os.getenv("SOCIALDATA_API_KEY", "").strip()

SOCIALDATA_BASE_URL = "https://api.socialdata.tools"
REQUEST_TIMEOUT = (10, 30)
MAX_TWEET_PAGES = 100
MAX_AVATAR_BYTES = 5 * 1024 * 1024

CARD_WIDTH = 800
CARD_HEIGHT = 450
BLUE = "#1D4ED8"
GRAY = "#657786"
LIGHT_GRAY = "#AAB8C2"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("x_impressions_bot")

user_sessions = {}


class BotWorkError(Exception):
    """Ошибка, которую можно безопасно и понятно показать пользователю."""


def api_get(path, params=None):
    """Выполняет GET-запрос к SocialData и возвращает JSON."""
    if not SOCIALDATA_API_KEY:
        raise BotWorkError(
            "Не задан SOCIALDATA_API_KEY. Добавьте ключ SocialData "
            "в переменные окружения и перезапустите бота."
        )

    response = requests.get(
        f"{SOCIALDATA_BASE_URL}{path}",
        headers={
            "Authorization": f"Bearer {SOCIALDATA_API_KEY}",
            "Accept": "application/json",
        },
        params=params or {},
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code == 401 or response.status_code == 403:
        raise BotWorkError("SocialData отклонил API-ключ. Проверьте его в настройках.")
    if response.status_code == 404:
        raise BotWorkError("Автор X не найден или SocialData не вернул профиль.")
    if response.status_code == 429:
        raise BotWorkError("SocialData ограничил частоту запросов. Попробуйте позже.")
    if response.status_code >= 500:
        raise BotWorkError("SocialData временно недоступен. Попробуйте позже.")
    if not response.ok:
        raise BotWorkError(f"SocialData вернул ошибку HTTP {response.status_code}.")

    try:
        return response.json()
    except ValueError as exc:
        raise BotWorkError("SocialData вернул ответ в неожиданном формате.") from exc


def get_user_profile(username):
    payload = api_get(f"/twitter/user/{username}")
    profile = payload.get("user", payload) if isinstance(payload, dict) else None
    if not isinstance(profile, dict):
        raise BotWorkError("Не удалось прочитать профиль автора в ответе SocialData.")

    user_id = profile.get("id") or profile.get("user_id")
    if user_id is None:
        raise BotWorkError("В ответе SocialData отсутствует ID автора.")

    avatar_url = (
        profile.get("profile_image_url_https")
        or profile.get("profile_image_url")
        or ""
    )
    if avatar_url:
        avatar_url = avatar_url.replace("_normal", "_400x400")

    return {
        "id": str(user_id),
        "name": str(profile.get("name") or username),
        "username": str(profile.get("screen_name") or username).lstrip("@"),
        "avatar_url": avatar_url,
    }


def extract_tweets(payload):
    """Поддерживает распространённые варианты обёртки списка твитов."""
    if isinstance(payload, list):
        return payload, None
    if not isinstance(payload, dict):
        raise BotWorkError("Список постов SocialData имеет неожиданный формат.")

    tweets = None
    for key in ("tweets", "data", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            tweets = value
            break
    if tweets is None:
        raise BotWorkError("В ответе SocialData не найден список постов.")

    cursor = payload.get("next_cursor")
    if cursor is None:
        cursor = payload.get("nextCursor")
    return tweets, cursor


def parse_created_at(value):
    if not value:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None

    text = str(value).strip()
    # Стандартная дата Twitter: "Wed Oct 10 20:19:24 +0000 2018".
    try:
        parsed = datetime.strptime(text, "%a %b %d %H:%M:%S %z %Y")
        return parsed
    except ValueError:
        pass

    # ISO-8601, включая варианты с Z и без часового пояса.
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return None


def read_views(tweet):
    value = tweet.get("views_count")
    if value is None:
        value = tweet.get("impression_count")
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def collect_tweet_stats(user_id):
    """Возвращает оригинальные посты как пары (дата, просмотры)."""
    rows = []
    cursor = None
    seen_cursors = set()

    for _ in range(MAX_TWEET_PAGES):
        params = {"cursor": cursor} if cursor else {}
        payload = api_get(f"/twitter/user/{user_id}/tweets", params=params)
        tweets, next_cursor = extract_tweets(payload)

        for tweet in tweets:
            if not isinstance(tweet, dict):
                continue
            # Не включаем ретвиты чужих публикаций.
            if "retweeted_status" in tweet:
                continue
            created_at = parse_created_at(tweet.get("created_at"))
            if created_at is None:
                continue
            rows.append((created_at, read_views(tweet)))

        if not next_cursor:
            break
        next_cursor = str(next_cursor)
        if next_cursor in seen_cursors or next_cursor == cursor:
            logger.warning("Остановка пагинации: повторился next_cursor")
            break
        seen_cursors.add(next_cursor)
        cursor = next_cursor
    else:
        logger.warning("Достигнут лимит страниц SocialData: %d", MAX_TWEET_PAGES)

    rows.sort(key=lambda item: item[0])
    if not rows:
        raise BotWorkError(
            "Не удалось найти оригинальные посты с датой публикации. "
            "Проверьте username и доступность данных в SocialData."
        )
    return rows


def format_number(value):
    return f"{int(value):,}".replace(",", " ")


def month_year(dt):
    months = (
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    )
    return f"{months[dt.month - 1]} {dt.year}"


def load_font(size, bold=False):
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
                "Arial Bold.ttf",
                "arialbd.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
                "Arial.ttf",
                "arial.ttf",
            ]
        )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except (OSError, ValueError):
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def download_avatar(avatar_url):
    """Скачивает только HTTPS-аватар с домена X; при сбое вернёт None."""
    if not avatar_url:
        return None
    try:
        parsed = urlparse(avatar_url)
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or not (
            host == "twimg.com" or host.endswith(".twimg.com")
        ):
            logger.warning("Пропущен аватар с неожиданного домена")
            return None

        with requests.get(
            avatar_url,
            timeout=REQUEST_TIMEOUT,
            stream=True,
            headers={"User-Agent": "Mozilla/5.0"},
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if not content_type.lower().startswith("image/"):
                return None
            data = bytearray()
            for chunk in response.iter_content(64 * 1024):
                if not chunk:
                    continue
                data.extend(chunk)
                if len(data) > MAX_AVATAR_BYTES:
                    logger.warning("Пропущен аватар больше допустимого размера")
                    return None
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return image
    except Exception:
        logger.exception("Не удалось загрузить аватар; будет использован запасной")
        return None


def make_cumulative(rows):
    dates = [row[0] for row in rows]
    cumulative = []
    total = 0
    for _, views in rows:
        total += views
        cumulative.append(total)
    return dates, cumulative, total


def smoothed_points(values, samples_per_segment=16):
    """Catmull-Rom curve with segment clipping to keep cumulative values monotonic."""
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return np.array([0.0, 1.0]), np.array([values[0], values[0]])

    out_x = []
    out_y = []
    for index in range(len(values) - 1):
        p0 = values[max(0, index - 1)]
        p1 = values[index]
        p2 = values[index + 1]
        p3 = values[min(len(values) - 1, index + 2)]
        t_values = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
        for t in t_values:
            y = 0.5 * (
                2 * p1
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t
            )
            y = min(max(y, min(p1, p2)), max(p1, p2))
            out_x.append(index + t)
            out_y.append(y)
    out_x.append(float(len(values) - 1))
    out_y.append(float(values[-1]))
    return np.asarray(out_x), np.asarray(out_y)


def create_chart(values, output_path):
    fig = plt.figure(figsize=(7.2, 1.55), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    x, y = smoothed_points(values)
    lower = min(0.0, float(np.min(y)))
    upper = max(1.0, float(np.max(y)))
    padding = (upper - lower) * 0.10
    ax.set_xlim(float(x[0]), max(float(x[-1]), float(x[0]) + 1))
    ax.set_ylim(lower, upper + padding)
    ax.fill_between(x, y, 0, color=BLUE, alpha=0.08, linewidth=0)
    ax.plot(x, y, color=BLUE, linewidth=2.5, solid_capstyle="round")
    fig.savefig(output_path, transparent=True, dpi=100, pad_inches=0)
    plt.close(fig)


def fit_text(draw, text, font, max_width):
    text = str(text)
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return text
    while text and draw.textbbox((0, 0), text + "…", font=font)[2] > max_width:
        text = text[:-1]
    return text + "…"


def create_card(profile, project_name, dates, cumulative, total):
    canvas = Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT), "#FFFFFF")
    draw = ImageDraw.Draw(canvas)

    avatar = download_avatar(profile.get("avatar_url"))
    avatar_size = 80
    if avatar is None:
        avatar = Image.new("RGB", (avatar_size, avatar_size), "#E5E7EB")
        avatar_draw = ImageDraw.Draw(avatar)
        initial = (profile.get("name") or "?")[:1].upper()
        initial_font = load_font(38, bold=True)
        box = avatar_draw.textbbox((0, 0), initial, font=initial_font)
        avatar_draw.text(
            ((avatar_size - (box[2] - box[0])) / 2, (avatar_size - (box[3] - box[1])) / 2 - box[1]),
            initial,
            font=initial_font,
            fill="#667085",
        )
    else:
        side = min(avatar.size)
        left = (avatar.width - side) // 2
        top = (avatar.height - side) // 2
        avatar = avatar.crop((left, top, left + side, top + side))
        avatar = avatar.resize((avatar_size, avatar_size), Image.Resampling.LANCZOS)

    mask = Image.new("L", (avatar_size, avatar_size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, avatar_size - 1, avatar_size - 1), fill=255)
    canvas.paste(avatar, (40, 35), mask)

    name_font = load_font(28, bold=True)
    username_font = load_font(20)
    number_font = load_font(72, bold=True)
    body_font = load_font(18)
    date_font = load_font(16)

    name = fit_text(draw, profile.get("name", ""), name_font, 610)
    draw.text((140, 39), name, font=name_font, fill="#111111")
    draw.text(
        (140, 77),
        f"@{profile.get('username', '').lstrip('@')}",
        font=username_font,
        fill=GRAY,
    )

    draw.text((40, 132), format_number(total), font=number_font, fill="#111111")
    project_name = re.sub(r"\s+", " ", str(project_name)).strip() or "Project"
    project_text = fit_text(
        draw,
        f"Impressions generated for {project_name}",
        body_font,
        CARD_WIDTH - 80,
    )
    draw.text((40, 222), project_text, font=body_font, fill=GRAY)

    temp_chart = None
    temp_card = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="temp_chart_", suffix=".png", delete=False
        ) as chart_file:
            temp_chart = chart_file.name
        with tempfile.NamedTemporaryFile(
            prefix="analytics_", suffix=".png", delete=False
        ) as card_file:
            temp_card = card_file.name

        create_chart(cumulative, temp_chart)
        chart = Image.open(temp_chart).convert("RGBA")
        canvas_rgba = canvas.convert("RGBA")
        canvas_rgba.alpha_composite(chart, (40, 250))
        canvas = canvas_rgba.convert("RGB")
        draw = ImageDraw.Draw(canvas)

        period = f"{month_year(dates[0])} - {month_year(dates[-1])}"
        draw.text((40, 414), period, font=date_font, fill=LIGHT_GRAY)
        canvas.save(temp_card, format="PNG", optimize=True)
        return temp_card, temp_chart
    except Exception:
        if temp_card and os.path.exists(temp_card):
            os.remove(temp_card)
        if temp_chart and os.path.exists(temp_chart):
            os.remove(temp_chart)
        raise


def process_request(chat_id, username, project_name, status_message_id):
    card_path = None
    chart_path = None
    try:
        profile = get_user_profile(username)
        rows = collect_tweet_stats(profile["id"])
        dates, cumulative, total = make_cumulative(rows)
        card_path, chart_path = create_card(
            profile, project_name, dates, cumulative, total
        )

        try:
            bot.delete_message(chat_id, status_message_id)
        except Exception:
            logger.info("Не удалось удалить промежуточное сообщение")

        with open(card_path, "rb") as photo:
            bot.send_photo(
                chat_id,
                photo,
                caption=(
                    f"Готово: {format_number(total)} просмотров "
                    f"по оригинальным постам @{profile['username']}."
                ),
            )
    except BotWorkError as exc:
        logger.info("Запрос не выполнен: %s", exc)
        try:
            bot.delete_message(chat_id, status_message_id)
        except Exception:
            pass
        bot.send_message(chat_id, f"⚠️ {exc}")
    except Exception:
        logger.exception("Непредвиденная ошибка при обработке запроса")
        try:
            bot.delete_message(chat_id, status_message_id)
        except Exception:
            pass
        bot.send_message(
            chat_id,
            "⚠️ Не удалось сформировать карточку. Попробуйте ещё раз позже.",
        )
    finally:
        for path in (card_path, chart_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    logger.warning("Не удалось удалить временный файл")


if not TG_TOKEN:
    raise SystemExit("Задайте TG_TOKEN в переменных окружения перед запуском.")

bot = telebot.TeleBot(TG_TOKEN, parse_mode=None)


@bot.message_handler(commands=["start"])
def start(message):
    user_sessions[message.from_user.id] = {"step": "username"}
    bot.reply_to(
        message,
        "Привет! Соберу статистику просмотров оригинальных постов X "
        "и подготовлю карточку.\n\n"
        "Шаг 1 из 2 — введи username автора в X без знака @.",
    )


@bot.message_handler(commands=["cancel"])
def cancel(message):
    user_sessions.pop(message.from_user.id, None)
    bot.reply_to(message, "Проверка отменена. Чтобы начать заново, отправь /start.")


@bot.message_handler(content_types=["text"])
def handle_text(message):
    user_id = message.from_user.id
    session = user_sessions.get(user_id)
    if not session:
        bot.reply_to(message, "Чтобы начать проверку, отправь /start.")
        return

    if session.get("step") == "username":
        username = message.text.strip().lstrip("@")
        if not re.fullmatch(r"[A-Za-z0-9_]{1,15}", username):
            bot.reply_to(
                message,
                "Укажи корректный username X: до 15 латинских букв, цифр "
                "или символов подчёркивания, без пробелов.",
            )
            return
        session["username"] = username
        session["step"] = "project"
        bot.reply_to(message, "Шаг 2 из 2 — введи название проекта.")
        return

    if session.get("step") == "project":
        project_name = message.text.strip()
        if not project_name:
            bot.reply_to(message, "Название проекта не должно быть пустым.")
            return
        if len(project_name) > 100:
            bot.reply_to(message, "Название проекта слишком длинное (максимум 100 символов).")
            return

        username = session["username"]
        session["step"] = "processing"
        status = bot.send_message(
            message.chat.id,
            "🔄 Запускаю проверку постов автора... Пожалуйста, подождите",
        )
        try:
            process_request(
                message.chat.id,
                username,
                project_name,
                status.message_id,
            )
        finally:
            user_sessions.pop(user_id, None)
        return

    if session.get("step") == "processing":
        bot.reply_to(message, "Запрос ещё обрабатывается. Дождись карточки или отправь /cancel.")


if __name__ == "__main__":
    logger.info("Telegram-бот запущен")
    bot.infinity_polling(skip_pending=True, timeout=30, long_polling_timeout=30)
