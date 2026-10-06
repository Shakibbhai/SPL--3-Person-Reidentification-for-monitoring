#!/bin/bash
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$REPO_DIR/backend"
PYTHON_BIN="/venv/main/bin/python"

if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

echo "=========================================================="
echo "DEPLOYING PERCEPTA-REID ON GPU VIA SUPERVISOR"
echo "Repository : $REPO_DIR"
echo "Backend    : $BACKEND_DIR"
echo "Python     : $PYTHON_BIN"
echo "Port       : 10100 (Public: 4062)"
echo "=========================================================="

# Stop and remove old personvit service if present
if [ -f "/etc/supervisor/conf.d/personvit.conf" ]; then
    echo "Stopping previous personvit service..."
    supervisorctl stop personvit 2>/dev/null || true
    rm -f /etc/supervisor/conf.d/personvit.conf
fi

# Write Percepta-ReID supervisor configuration
SUPERVISOR_CONF="/etc/supervisor/conf.d/percepta.conf"
cat << EOF > "$SUPERVISOR_CONF"
[program:percepta]
directory=$BACKEND_DIR
command=$PYTHON_BIN main.py
environment=PORT="10100",HOST="0.0.0.0"
autostart=true
autorestart=true
stopasgroup=true
killasgroup=true
stopsignal=TERM
stdout_logfile=/workspace/percepta_app.log
redirect_stderr=true
EOF

# Ensure GitHub self-hosted runner stays active in supervisor
if [ -d "/workspace/actions-runner" ]; then
    cat << EOF > /etc/supervisor/conf.d/github-runner.conf
[program:github-runner]
directory=/workspace/actions-runner
command=/bin/bash -c "RUNNER_ALLOW_RUNASROOT=1 ./run.sh"
autostart=true
autorestart=true
stopasgroup=true
killasgroup=true
stdout_logfile=/workspace/runner.log
redirect_stderr=true
EOF
fi

supervisorctl reread
supervisorctl update

echo "Restarting percepta service..."
supervisorctl restart percepta || supervisorctl start percepta

sleep 4

STATUS=$(supervisorctl status percepta)
echo "Current Percepta Status: $STATUS"

if echo "$STATUS" | grep -q "RUNNING"; then
    echo "✅ Percepta-ReID successfully deployed and running!"
    echo "Logs: /workspace/percepta_app.log"
    echo "Internal: http://0.0.0.0:10100"
    echo "Public  : http://101.108.153.134:4062"
else
    echo "❌ Deployment status abnormal. Log excerpt:"
    tail -n 30 /workspace/percepta_app.log
    exit 1
fi
