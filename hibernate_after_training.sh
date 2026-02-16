#!/bin/bash
# Runs a training command, then hibernates after a 5-minute cancellable countdown.
# Usage: bash hibernate_after_training.sh python -m primary_model_training.train [args...]
# Press Ctrl+C during the countdown to cancel hibernation.

if [ $# -eq 0 ]; then
    echo "Usage: bash hibernate_after_training.sh <training command...>"
    echo "Example: bash hibernate_after_training.sh python -m primary_model_training.train --epochs 100"
    exit 1
fi

# Run the training command
echo "=== Starting training ==="
"$@"
EXIT_CODE=$?
echo ""
echo "=== Training finished (exit code: $EXIT_CODE) ==="

if [ $EXIT_CODE -ne 0 ]; then
    echo "Training failed. Skipping hibernation."
    exit $EXIT_CODE
fi

# Countdown with cancellation
SECONDS_LEFT=300
CANCELLED=false

trap 'CANCELLED=true' INT

echo "Hibernating in 5 minutes. Press Ctrl+C to cancel."
echo ""

while [ $SECONDS_LEFT -gt 0 ] && [ "$CANCELLED" = false ]; do
    MINS=$((SECONDS_LEFT / 60))
    SECS=$((SECONDS_LEFT % 60))
    printf "\rHibernating in %d:%02d... " $MINS $SECS
    sleep 1
    SECONDS_LEFT=$((SECONDS_LEFT - 1))
done

trap - INT
echo ""

if [ "$CANCELLED" = true ]; then
    echo "Hibernation cancelled."
    exit 0
fi

echo "Hibernating now..."
shutdown /h
