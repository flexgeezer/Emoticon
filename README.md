# Emoticon

Real-time emotion recognition desktop app — detects **Happy, Sad, Confused, Angry** from a webcam or video file.

Built with Python, TensorFlow, OpenCV, and Tkinter.

## Modes

- **Live** — emoji + colour overlay in real time
- **Diagnosis** — 60-second session, saves a report with emotion graph and comments
- **Report History** — browse past reports

## Setup & Run
Need to install Tkinter beforehand

```bash
pip install -r requirements.txt
python model/emoticonmodel.py
python app.py
```

## Model

MobileNetV2 backbone fine-tuned for 4 emotion classes. Lives in `model/`. To retrain, update the dataset and re-run `model/emoticonmodel.py`.
