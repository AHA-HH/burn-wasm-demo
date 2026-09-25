#!/usr/bin/env bash

# Opening index.html directly (file://) fails: modules and fetch() of the
# .wasm both require an http(s) origin. Hence a local server.

if ! command -v python3 &> /dev/null
then
    echo "python3 could not be found. Running server requires python3."
    exit
fi

echo "Running local python HTTP server on port 8000 ..."
python3 -m http.server 8000
