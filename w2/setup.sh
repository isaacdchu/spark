#!/bin/bash

DIR="$(dirname "$0")"

mkdir -p "$DIR"/build
mkdir -p "$DIR"/include

if [ -d "$DIR"/include/taskflow/ ]; then
    echo "Taskflow already exists, skipping clone."
else
    echo "Cloning taskflow..."
    mkdir -p "$DIR"/include/taskflow
    cd "$DIR" && git clone "https://github.com/taskflow/taskflow.git"
    echo "Copying taskflow headers..."
    cp -r taskflow/taskflow/ include/
    echo "Cleaning up taskflow clone..."
    rm -rf taskflow/
fi

echo "Setup complete."
