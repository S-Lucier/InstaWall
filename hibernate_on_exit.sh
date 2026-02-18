#!/bin/bash
# Wait for training PID to exit, then hibernate
PID=33132
echo "Watching PID $PID for exit..."
while true; do
    if ! tasklist.exe //FI "PID eq $PID" 2>/dev/null | grep -q "$PID"; then
        echo "PID $PID exited"
        break
    fi
    sleep 30
done
echo "Hibernating now..."
cmd.exe //C "shutdown /h"
