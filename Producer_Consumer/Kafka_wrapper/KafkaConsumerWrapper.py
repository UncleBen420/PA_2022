from confluent_kafka import Consumer
import threading

class KafkaConsumerWrapper:

    def __init__(self, address, port, group_id):
        self.address = address
        self.port = port
        self.group_id = group_id
        
    def connect(self):
        """connect to the broker"""
        connection_str = self.address + ":" + self.port
        conf = {
        'bootstrap.servers' : connection_str,
        'group.id' : self.group_id,
        'auto.offset.reset': 'earliest'
        }
        print("connecting to Kafka topic...")
        self.consumer = Consumer(conf)

    def listen(self):
        self.consumer.subscribe([self.listener_topic])
        while not self.shutdown_flag:
            msg = self.consumer.poll(0.5)
            if msg is None:
                continue
            if msg.error():
                print("Consumer error: {}".format(msg.error()))
                continue
            self.listener_callback(msg.value())
        self.consumer.close()
    
    def subscribe(self, topic, callback):
        """subscribe on a topic and listen to upcoming messages"""
        self.listener_topic = topic
        self.listener_callback = callback
        self.shutdown_flag = False
        self.listener = threading.Thread(target=self.listen)
        # to ensure the thread will not be terminated at the wrong time
        self.listener.setDaemon(True)
        self.listener.start()
        
    def unsubscribe(self):
        self.shutdown_flag = True
        self.listener.join()
        self.listener = None
    
    def disconnect(self):
        """kafka doesn't need to disconnect but to respect the interface the method is implemented"""
        pass
