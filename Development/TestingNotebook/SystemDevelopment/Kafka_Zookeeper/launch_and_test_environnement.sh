#!/bin/sh

# https://medium.com/rahasak/kafka-and-zookeeper-with-docker-65cff2c2c34f

# Start containers
docker-compose up -d --remove-orphans

sleep 5

echo "\nensure the containers are running"
docker ps -a

sleep 5 # to ensure kafka has properly launch

echo '\nlist the topic (should be empty)'
docker exec kafkaPA kafka-topics --bootstrap-server kafka:29092 --list

echo '\nCreating kafka topics'
docker exec kafkaPA kafka-topics --create --if-not-exists --bootstrap-server kafka:29092 --partitions 1 --replication-factor 1 --topic topicTest

echo '\nlist the topic (should not be empty)'
docker exec kafkaPA kafka-topics --bootstrap-server kafka:29092 --list

echo "\nCLIENT TEST\n"

echo "\nensure client can see topics"
kafkacat -L -b localhost:29092
echo "\npublish hello world over topic topicTest"
echo "hello world!" | kafkacat -P -b localhost:29092 -t topicTest
echo "\nlisten on the topic topicTest"
kafkacat -C -b localhost:29092 -t topicTest



