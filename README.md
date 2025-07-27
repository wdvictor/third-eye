
<p align="center">
  <img src="third-eye-logo.png" alt="Logo Third Eye" width="300"/>
</p>


# 🎥 Motion Detection System

This Python project uses a webcam to detect movement, triggering a real-time alert system that:

- 🔔 Emits a **sound alarm** (optional)
- 🎞️ **Records videos** of detected motion
- 🖼️ **Captures images**
- 📝 **Logs events** with timestamp
- 🧑 **Face detection and recogniton**

The system is designed for **surveillance and monitoring** and can be extended for more advanced computer vision tasks.

## 🚀 Features

- Motion detection using OpenCV
- Alarm sound triggered on detection (can be silenced)
- Video and image capture with timestamps
- Logs saved to a text file
- Auto-creation of output folders

### 🔮 Upcoming Features

- Face detection
- shadow-detection

## 🛠️ System Requirements

Before installing Python libraries, make sure the following system dependencies are installed:

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install sox 
sudo apt install libsox-fmt-all
sudo apt install libasound2-dev
sudo apt install build-essential cmake
sudo apt install libopenblas-dev liblapack-dev
sudo apt install libx11-dev libgtk-3-dev
sudo apt install python3-dev

```

## 📦 Installation

```bash
git clone https://github.com/wdvictor/third-eye.git
cd third-eye
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
pip3 install git+https://github.com/ageitgey/face_recognition_models
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
  Valid options currently include: **`video-only`**, **`image-only`**, **`face-only`**.

- **`--silent`**  
  Disables the alarm sound when passed.


                    
            

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