#!/bin/bash
# run_server.sh

# Create necessary directories if they don't exist
mkdir -p player_data

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the server
echo "Starting Treasure Hunter Game Server..."
python main.py