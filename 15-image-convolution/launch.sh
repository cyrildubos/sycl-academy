#!/bin/bash

icpx -fsycl -o main -I include -I include/stb src/main.cpp

./main