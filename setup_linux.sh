#!/bin/bash
# Setup-Skript für YOLOv26 auf Ubuntu 20.04
# Ausführen mit: bash setup_linux.sh

set -e  # Bei Fehler abbrechen

echo "=== YOLOv26 Linux Setup ==="

# 1. Systemabhängigkeiten installieren (benötigt sudo)
echo ""
echo "[1/4] Systemabhängigkeiten installieren..."
sudo apt-get update -q
sudo apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libqt5widgets5 \
    python3-pyqt5

# 2. Virtuelle Umgebung erstellen
echo ""
echo "[2/4] Virtuelle Python-Umgebung erstellen..."
python3 -m venv yolov26
source yolov26/bin/activate

# 3. pip aktualisieren und Pakete installieren
echo ""
echo "[3/4] Python-Pakete installieren..."
pip install --upgrade pip
pip install -r requirements_linux.txt

# 4. Kamerazugriff einrichten (Benutzer zur video-Gruppe hinzufügen)
echo ""
echo "[4/4] Benutzer zur video-Gruppe hinzufügen..."
sudo usermod -aG video "$USER"

echo ""
echo "=== Setup abgeschlossen! ==="
echo ""
echo "Virtuelle Umgebung aktivieren mit:"
echo "  source yolov26/bin/activate"
echo ""
echo "HINWEIS: Bitte neu einloggen oder 'newgrp video' ausführen,"
echo "damit der Kamerazugriff wirksam wird."
