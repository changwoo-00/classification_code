import sys
import pika
import cv2
import numpy as np
import queue
import threading
import time

class RabbitMQ_producer(threading.Thread):
    def __init__(self, name, host='localhost', port=5672, exchange_name='my_exchange_1', exchange_type='direct', routing_key='my_key_1'):
        threading.Thread.__init__(self)
        self.name = name
        self.host = host
        self.port = port
        self.exchange_name = exchange_name
        self.exchange_type = exchange_type
        self.routing_key = routing_key
        self.channel = None
        self.queue = queue.Queue(10)        # internal message queue. Note that Queue is thread-safe.
        self.do_exit = False

    def run(self):  
        try:      
            # establish connection with the RabbitMQ server
            print(" [%s] Connecting to RabbitMQ server..." % self.name)
            parameters = pika.ConnectionParameters(host=self.host, port=self.port)
            connection = pika.BlockingConnection(parameters)
            self.channel = connection.channel()

            # declare the exchange
            self.channel.exchange_declare(exchange=self.exchange_name, exchange_type=self.exchange_type)

            print(' [%s] routing key: %s' % (self.name, self.routing_key))

            # waiting & sending messages
            try:
                while self.do_exit == False:
                    try:
                        msg = self.queue.get(block=True, timeout=1)
                        print(" [%s] Sending a message of length %d" % (self.name, len(msg)))
                        self.channel.basic_publish(exchange=self.exchange_name, routing_key=self.routing_key, body=msg)
                    except queue.Empty:
                        time.sleep(1)
            except BaseException as e:
                print(" [%s] An exception or interrupt occurred! %s" % (self.name, e))
            
            # shutdown
            print(" [%s] Closing down connections..." % self.name)
            #channel.stop_consuming()        
            connection.close()
        except BaseException as e:
            print(" [%s] A major exception has occurred!! %s" % (self.name, e))

        print(" [%s] Finished" % self.name)

    def send(self, msg):
        if self.isAlive() == False:
            raise Exception(' [%s] thread not running!' % self.name)
            
        print(" [%s] Queuing a message of size %d : %s" % (self.name, len(msg), msg))
        self.queue.put(msg)

    def send_image(self, img, encoding_format='.png'):
        if self.isAlive() == False:
            raise Exception(' [%s] thread not running!' % self.name)
            
        print(" [%s] Queuing an image of size %dx%d" % (self.name, img.shape[0], img.shape[1]))
        img_str = cv2.imencode(encoding_format, img)[1].tostring()
        self.queue.put(img_str)

    def send_EOT(self):
        if self.isAlive() == False:
            raise Exception(' [%s] thread not running!' % self.name)       
        print(" [%s] Queuing EOT" % self.name)
        self.queue.put(b'\x04')

    def exit(self):
        print(" [%s] Setting exit flag" % self.name)
        self.do_exit = True

    def flush_and_exit(self):
        print(" [%s] Waiting for message queue to flush..." % self.name)
        while not self.queue.empty() and self.isAlive():
            time.sleep(1)
        print(" [%s] Setting exit flag" % self.name)
        self.do_exit = True