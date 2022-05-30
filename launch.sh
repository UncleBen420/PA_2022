#!/bin/bash

pip install -r requirements.txt
cd Final/KafkaZookeeper/
./launch_and_test_environnement.sh
cd ..
python3 system_integration_test.py
cd KafkaZookeeper/
docker-compose down
