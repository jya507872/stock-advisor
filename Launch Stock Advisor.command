#!/usr/bin/env bash
# Double-click this file in Finder to start Stock Advisor and open it in your browser.

DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=8000
URL="http://localhost:$PORT"

echo "=== Stock Advisor ==="
echo "Directory: $DIR"
echo ""

# Check if something is already on the port
if lsof -i :"$PORT" -sTCP:LISTEN -t &>/dev/null; then
  echo "Server already running on port $PORT — opening browser."
  open "$URL"
  exit 0
fi

# Prefer the venv python if one exists, then python3
if   [ -x "$DIR/.venv/bin/python3" ]; then PYTHON="$DIR/.venv/bin/python3"
elif command -v python3 &>/dev/null;  then PYTHON="python3"
else echo "Error: python3 not found."; read -n1 -p "Press any key to close…"; exit 1
fi

echo "Python: $PYTHON"
echo "Installing / verifying dependencies…"
"$PYTHON" -m pip install -q --break-system-packages \
  yfinance pandas numpy rich requests fastapi uvicorn 2>&1 | tail -3

echo ""
echo "Starting server on $URL …"
cd "$DIR"
"$PYTHON" app.py &
SERVER_PID=$!

# Wait until the server accepts connections (up to 15 s)
for i in $(seq 1 30); do
  sleep 0.5
  if curl -s -o /dev/null "$URL"; then
    break
  fi
done

echo "Opening browser…"
open "$URL"

echo ""
echo "Server is running (PID $SERVER_PID). Close this window to stop it."
wait "$SERVER_PID"
