# 🖥️ AI Expert System for Workplace Activity & Productivity Monitoring

📋 *[Читать на русском языке](README_RU.md)*

---

An intelligent, real-time Computer Vision expert system that monitors employee workspace presence, tracks engagement levels, analyzes emotional states, and computes dynamic productivity scores. The system processes a live video stream, processes state machine rule shifts, manages historical logs, and sends instant automated HR alerts/reports via Telegram.

## 🛠️ Architecture & Tech Stack

The core engine is built using standard non-blocking loops combined with state-of-the-art Deep Learning frameworks:

* **Object Detection Stack:** `YOLOv8` (Ultralytics nano model `yolov8n.pt`) — optimized for high-velocity real-time human and equipment ("person", "laptop") boundary detection.
* **Facial Emotion Recognition:** `DeepFace` — pipeline executing convolutional layers to determine the dominant emotional state (updated every 10 seconds to prevent thread blocking).
* **Data Processing & Storage:** `Pandas` — drives logging across multiple local structured schemas (`presence_logs`, `event_logs`, `score_logs`).
* **Communication Gateway:** `Telegram Bot API` — delivers live threshold violation alerts and system end-of-day analytics.

## 🚀 Expert Logic & Scoring Engine

The internal algorithmic rule engine monitors parameters every second and controls a real-time state machine:

### ⚙️ Activity State Tree
* **`WORKER_PRESENT`** 🟢 Employee and working machine (laptop) are visible simultaneously. Triggers active work events and gives a reward bonus (`+0.1` points/min).
* **`NONWORKING`** 🟡 Employee is detected at the workspace without their machine. Triggers passive idle event and applies a penalty (`-0.1` points/min).
* **`ABSENT`** 🔴 Workspace is completely empty. Triggers absence event and decays core score (`-0.2` points/min).

### ⚠️ Automated Infraction Alerts
If the system tracks an uninterrupted **`ABSENT`** state exceeding `120 seconds` (2 minutes), it instantly executes a direct high-priority critical warning pushed to the supervisor via Telegram Bot and applies a score reduction penalty of `-5.0` points.

## 📸 Application Interface / Скриншоты

<details>
<summary><b>Click to expand / Нажмите, чтобы открыть скриншоты</b></summary>

### 📈 Computer Vision Real-Time Monitoring
<img src="expert_monitor.png" alt="Expert Monitor Dashboard" width="750" />

### 📅 Automated HR Discipline Report via Telegram
<img src="expert_report.png" alt="Telegram Report Output" width="450" />

</details>

---

## 💻 Local Setup & Execution

1. Clone this repository:
   ```bash
   git clone https://github.com
   cd employee-activity-expert-system
   ```

2. Install the required deep learning and data stack:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have a web-camera connected and your `.env` file configured, then boot the system:
   ```bash
   python monitor.py
   ```
