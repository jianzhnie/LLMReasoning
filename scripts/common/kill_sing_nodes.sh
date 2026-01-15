#!/bin/bash

echo "Kill all tasks with the specified token"

if [ $# -eq 0 ]; then
    echo "Help: please specify the related token"
    echo "First parameter: the filtered string."
    echo "Second parameter: excluding the specified string (optional)."
    exit 1
elif [ $# -eq 1 ]; then
    ps aux | grep "$1" | grep -v grep
elif [ $# -eq 2 ]; then
    ps aux | grep "$1" | grep -v grep | grep -v "$2"
else
    echo "Please specify one or two related tokens"
    exit 1
fi

read -s -n1 -p "Are you sure you want to kill them? (y/n) " pressed_key
echo

if [ "$pressed_key" = "y" ]; then
    if [ $# -eq 1 ]; then
        ps aux | grep "$1" | grep -v grep | awk '{print $2}' | xargs kill -9
    elif [ $# -eq 2 ]; then
        ps aux | grep "$1" | grep -v grep | grep -v "$2" | awk '{print $2}' | xargs kill -9
    fi
    echo "Processes killed."
else
    echo "Operation canceled."
fi
