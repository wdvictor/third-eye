# 🎥 Motion Detection System

This Python project uses a webcam to detect movement, triggering a real-time alert system that:

- 🔔 Emits a **sound alarm** (optional)
- 🎞️ **Records videos** of detected motion
- 🖼️ **Captures images**
- 📝 **Logs events** with timestamp

The system is designed for **surveillance and monitoring** and can be extended for more advanced computer vision tasks.

## 🚀 Features

- Motion detection using OpenCV
- Alarm sound triggered on detection (can be silenced)
- Video and image capture with timestamps
- Logs saved to a text file
- Auto-creation of output folders

### 🔮 Upcoming Features

- Face and vehicle detection
- License plate recognition
- Web-based monitoring dashboard
- Telegram notifications
- shadow-detection

## 📦 Installation

```bash
git clone https://github.com/wdvictor/third-eye.git
cd third-eye
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## 📷 Camera Setup

When the script starts, it will scan for available cameras.  
If there's more than one, it will prompt you to choose.

## ▶️ Running the Program

```bash
python main.py [OPTIONS]
```

### 🔧 Command-Line Options

- **`--modes`**  
  Specifies which functionalities to execute. Supports multiple values.  
  If omitted, all modes will be enabled by default.  
  Valid options currently include: **`video-only`**, **`image-only`**.

- **`--silent`**  
  Disables the alarm sound when passed.


                           |
                 

Example:

```bash
python main.py --mode image-only --silent
```

This will run the motion detector saving only images and skipping the alarm sound.

## 📂 Output

- `detected_images/` – stores captured images with timestamps  
- `detected_videos/` – stores 5-second video clips after each detection  
- `motion_detection_log.txt` – logs all motion events with filenames and timestamps




## Possible Issues

⚠️ Problem: **Alarm sound not playing – `play WARN alsa: can't encode 0-bit Unknown or not applicable`**

**Description**:  
When the alarm is triggered, the following warning appears, and no sound is played:

```
play WARN alsa: can't encode 0-bit Unknown or not applicable
```

**Cause**:  
This warning typically means that the `play` command (from the `sox` package) is installed, but your Linux system cannot play audio through ALSA (Advanced Linux Sound Architecture), which `play` uses by default. It may be missing audio codecs or the necessary permissions to access the audio device.

**Solutions**:

1. ✅ **Install `sox` with MP3 and ALSA support** (if not already installed):

   ```bash
   sudo apt install sox libsox-fmt-all
   ```

## 📄 License

This project is licensed under the [MIT License](LICENSE).