#!/bin/bash
# macOS launcher: double-click in Finder or run ./AVVIA_MAC.command [options]
cd "$(dirname "$0")" || exit 1

VENV_DIR=".venv-local"
VENV_PY="$VENV_DIR/bin/python"

fail() {
  echo
  echo "Não foi possível iniciar o Vanish Lab. Veja o README.md para soluções comuns."
  read -r -p "Aperte ENTER para fechar..."
  exit 1
}

if [ ! -x "$VENV_PY" ]; then
  if ! command -v python3 >/dev/null 2>&1; then
    echo "[Vanish Lab] Instale o Python 3 (https://www.python.org/downloads/macos/ ou 'brew install python')."
    fail
  fi
  echo "[Vanish Lab] Criando o ambiente Python local..."
  python3 -m venv "$VENV_DIR" || fail
fi

if ! "$VENV_PY" -c "import cv2, mediapipe, numpy, pyvirtualcam" >/dev/null 2>&1; then
  echo "[Vanish Lab] Instalando os componentes. Só precisa de internet desta vez..."
  "$VENV_PY" -m pip install --upgrade pip || fail
  "$VENV_PY" -m pip install -r requirements.txt || fail
fi

echo "[Vanish Lab] Abrindo a webcam..."
"$VENV_PY" vanish_cam.py "$@" || fail
