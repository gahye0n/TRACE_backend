#!/bin/bash
# 전람회 당일 시연용: 로컬 백엔드를 띄우고 Cloudflare Tunnel로 공개 HTTPS 주소를 뚫은 뒤,
# 프론트엔드(web-frontend/index.html)의 API_BASE를 그 주소로 바꿔 GitHub Pages에 push한다.
#
# 사용법: 이 스크립트가 있는 web-backend 폴더에서 실행
#   ./run_exhibition_demo.sh
#
# 종료하려면 Ctrl+C. 종료 시 백엔드·터널 프로세스를 함께 정리한다.

set -euo pipefail
cd "$(dirname "$0")"

FRONTEND_DIR="../web-frontend"
FRONTEND_URL="https://gahye0n.github.io/TRACE_frontend/"
PORT=8000

echo "[1/4] 백엔드(FastAPI) 로컬 실행 중..."
source venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port "$PORT" > /tmp/exhibition_backend.log 2>&1 &
BACKEND_PID=$!

cleanup() {
  echo
  echo "종료 중... 백엔드/터널 프로세스를 정리합니다."
  kill "$BACKEND_PID" 2>/dev/null || true
  [ -n "${TUNNEL_PID:-}" ] && kill "$TUNNEL_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "    -> 모델 로딩 대기 중..."
for i in $(seq 1 30); do
  if curl -s -o /dev/null "http://127.0.0.1:$PORT/api/health"; then
    echo "    -> 백엔드 준비 완료 (PID $BACKEND_PID)"
    break
  fi
  sleep 1
done

echo "[2/4] Cloudflare Tunnel 여는 중..."
cloudflared tunnel --url "http://127.0.0.1:$PORT" > /tmp/exhibition_tunnel.log 2>&1 &
TUNNEL_PID=$!

TUNNEL_URL=""
for i in $(seq 1 30); do
  TUNNEL_URL=$(grep -oE "https://[a-zA-Z0-9-]+\.trycloudflare\.com" /tmp/exhibition_tunnel.log | head -1 || true)
  [ -n "$TUNNEL_URL" ] && break
  sleep 1
done

if [ -z "$TUNNEL_URL" ]; then
  echo "터널 주소를 찾지 못했습니다. /tmp/exhibition_tunnel.log 를 확인하세요."
  exit 1
fi
echo "    -> 터널 주소: $TUNNEL_URL"

echo "[3/4] 프론트엔드 API_BASE 갱신 후 push..."
sed -i '' "s#^const API_BASE = \".*\";#const API_BASE = \"$TUNNEL_URL\";#" "$FRONTEND_DIR/index.html"
( cd "$FRONTEND_DIR" \
  && git add index.html \
  && git commit -q -m "Point API_BASE at today's exhibition tunnel URL" \
  && git push origin main )

echo "[4/4] 완료! 몇 분 내로 GitHub Pages에 반영됩니다."
echo
echo "  발표용 접속 주소: $FRONTEND_URL"
echo
echo "이 터미널을 켜두세요 (Ctrl+C로 종료). 종료하면 시연 사이트가 더 이상 동작하지 않습니다."
wait
