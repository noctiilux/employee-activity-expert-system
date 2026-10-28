import cv2
import time
import pandas as pd
import threading
from datetime import datetime, timedelta
from ultralytics import YOLO
from deepface import DeepFace
import requests

# ===================== CONFIG ===================== #
import os

if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            if "=" in line:
                key, val = line.strip().split("=", 1)
                os.environ[key] = val

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_BOT_TOKEN_HERE")
CHAT_ID = os.getenv("CHAT_ID", "YOUR_CHAT_ID_HERE")

LOG_FILE = "presence_log.csv"
EVENTS_FILE = "events_log.csv"
SCORE_FILE = "score_log.csv"


CHECK_INTERVAL = 1        # Проверка каждую секунду
EMOTION_INTERVAL = 10     # Обновление эмоции каждые 10 сек
ABSENT_THRESHOLD = 120    # 2 минуты
WORK_START = 9
WORK_END = 20
LUNCH_START = 12
LUNCH_END = 13

# Цвета для отрисовки
COLOR_PERSON = (0, 255, 0)
COLOR_LAPTOP = (255, 255, 0)

# ===================== INIT ===================== #
employee_name = input("Введите имя сотрудника: ").strip()
if not employee_name:
    employee_name = "Worker1"

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

last_emotion_time = datetime.now()
current_emotion = "unknown"
absent_start = None
last_status_minute = None

status_summary = {"WORKER_PRESENT": 0, "NONWORKING": 0, "ABSENT": 0}
emotion_summary = {}
score = 100  # стартовое значение

# ===================== FUNCTIONS ===================== #
def send_telegram(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": message})
    except Exception as e:
        print("Telegram Error:", e)

def analyze_emotion(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        return result[0]['dominant_emotion']
    except Exception:
        return "unknown"

def log_status(status, persons, laptops, emotion, score):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "Timestamp": timestamp,
        "Employee": employee_name,
        "Status": status,
        "Persons": persons,
        "Laptops": laptops,
        "Emotion": emotion,
        "Score": score
    }
    df = pd.DataFrame([row])
    df.to_csv(f"{employee_name}_{LOG_FILE}", mode='a', header=not pd.io.common.file_exists(f"{employee_name}_{LOG_FILE}"), index=False)
    print(f"[{timestamp}] {employee_name} | Status: {status} | Emotion: {emotion} | Score: {score}")

def log_event(event_text):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([{"Timestamp": timestamp, "Employee": employee_name, "Event": event_text}])
    df.to_csv(f"{employee_name}_{EVENTS_FILE}", mode='a', header=not pd.io.common.file_exists(f"{employee_name}_{EVENTS_FILE}"), index=False)
    print(f"📝 Event: {event_text}")

def log_score_change(reason, delta, score):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([{
        "Timestamp": timestamp,
        "Employee": employee_name,
        "Reason": reason,
        "Delta": delta,
        "New_Score": score
    }])
    df.to_csv(f"{employee_name}_{SCORE_FILE}", mode='a', header=not pd.io.common.file_exists(f"{employee_name}_{SCORE_FILE}"), index=False)

def generate_daily_report():
    total = sum(status_summary.values())
    if total == 0:
        return "Нет данных за день."

    work_ratio = status_summary["WORKER_PRESENT"] / total * 100
    nonwork_ratio = status_summary["NONWORKING"] / total * 100
    absent_ratio = status_summary["ABSENT"] / total * 100
    top_emotion = max(emotion_summary, key=emotion_summary.get) if emotion_summary else "unknown"

    recommendation = ""
    if absent_ratio > 20:
        recommendation = "⚠️ Частые отсутствия. Следует улучшить дисциплину."
    elif nonwork_ratio > 30:
        recommendation = "💻 Снижение активности. Возможно, требуется повышение мотивации."
    elif work_ratio > 70:
        recommendation = "✅ Отличная вовлеченность. Так держать!"
    else:
        recommendation = "📊 Стабильная работа, можно добавить фокус."

    report = (
        f"📅 Отчёт за день для {employee_name}\n"
        f"Рабочее время: {work_ratio:.1f}%\n"
        f"Без работы: {nonwork_ratio:.1f}%\n"
        f"Отсутствие: {absent_ratio:.1f}%\n"
        f"Основная эмоция: {top_emotion}\n"
        f"Итоговый балл: {score}\n"
        f"{recommendation}"
    )
    send_telegram(report)
    print("\n" + report)

# ===================== MAIN LOOP ===================== #
print("▶️ Система запущена. Нажмите Q для выхода.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = datetime.now()
    hour = now.hour

    # YOLO DETECTION
    results = model(frame, verbose=False)
    boxes = results[0].boxes
    persons = 0
    laptops = 0

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if label == "person" and conf > 0.5:
            persons += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PERSON, 2)
            cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PERSON, 2)
        if label == "laptop" and conf > 0.5:
            laptops += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_LAPTOP, 2)
            cv2.putText(frame, "Laptop", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_LAPTOP, 2)

    # Определение статуса
    if LUNCH_START <= hour < LUNCH_END:
        status = "LUNCH_BREAK"
    elif hour < WORK_START or hour >= WORK_END:
        status = "OFF_HOURS"
    elif persons == 0 and laptops == 0:
        status = "ABSENT"
    elif persons >= 1 and laptops == 0:
        status = "NONWORKING"
    elif persons >= 1 and laptops >= 1:
        status = "WORKER_PRESENT"
    else:
        status = "UNKNOWN"

    # События
    if status == "WORKER_PRESENT":
        log_event("👨‍💻 Работник активно работает")
    elif status == "NONWORKING":
        log_event("💤 Работник на месте, но не работает")
    elif status == "ABSENT":
        log_event("🚶 Работник отсутствует")

    # Проверка отсутствия
    if status == "ABSENT":
        if absent_start is None:
            absent_start = now
        elif (now - absent_start).seconds >= ABSENT_THRESHOLD:
            send_telegram("⚠️ Работник отсутствует более 2 минут!")
            score -= 5
            log_score_change("Отсутствие более 2 минут", -5, score)
            absent_start = None
    else:
        absent_start = None

    # Обновление эмоций
    if (now - last_emotion_time).seconds >= EMOTION_INTERVAL:
        current_emotion = analyze_emotion(frame)
        emotion_summary[current_emotion] = emotion_summary.get(current_emotion, 0) + 1
        last_emotion_time = now

    # Логирование раз в минуту
    if last_status_minute != now.minute:
        status_summary[status] = status_summary.get(status, 0) + 1
        log_status(status, persons, laptops, current_emotion, score)
        last_status_minute = now.minute

    # Бонусы
    if status == "WORKER_PRESENT":
        score += 0.1  # небольшой бонус
    elif status == "NONWORKING":
        score -= 0.1
    elif status == "ABSENT":
        score -= 0.2

    # Визуализация
    cv2.putText(frame, f"Status: {status}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Emotion: {current_emotion}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Score: {score:.1f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    cv2.imshow(f"Office Monitor - {employee_name}", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Итоговый отчёт
generate_daily_report()
