from confluent_kafka import Producer

class KafkaProducerWrapper:

    def __init__(self, address, port):
        self.address = address
        self.port = port
        
    def connect(self):
        """connect to the broker"""
        connection_str = self.address + ":" + self.port
        conf = {
        'bootstrap.servers' : connection_str
        }
        print("connecting to Kafka topic...")
        self.producer = Producer(conf)
        
    def init_message(self, frames_descriptor):
        pass

    def publish(self, topic, frame, callback):
        """publish to others"""
        self.producer.produce(topic=topic, value=frame, key=bytes(), on_delivery=callback)
        
    def flush(self):
        self.producer.flush()
        
    def disconnect(self):
        """kafka doesn't need to disconnect but to respect the interface the method is implemented"""
        pass
