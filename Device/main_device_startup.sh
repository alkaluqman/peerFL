#!/bin/bash
mkdir -p ./data
mkdir -p ./data/dev_0 ./data/dev_1 ./data/dev_2 ./data/dev_3 ./data/dev_4 ./data/dev_5 ./data/dev_6 ./data/dev_7 ./data/dev_controller
chmod "./data/dev_controller" 777

docker compose up --force-recreate --build dev_0 dev_1 dev_2 dev_3 dev_4 dev_5 dev_6 dev_7 dev_controller
