# for connecting & interacting w RabbitMQ (server chat client functions)
import pika
import json  # for parsing and creating JSON data
import threading  # for threading (concurrent processing DUH)
import os  # for OS interface for env variables
import dotenv  # for loading env variables from the .env file
from queue import Queue  # for using queue DS for message passing btw threads

# the following imports are from .py files that define methods and classes for doing these functions
# pretty self explanatory names

from font.fontAgent import getTitlefont
from highlight.highlight import getHighlight1
from image_process.title_location import getTitlePosition1
from image_process.logo_location import getLogoPosition1
from image_process.logo_color_selection import getLogoColor1
from image_process.title_logo_location import getTitleLogoPosition
from color.colorAgent import getTitlecolor1

# this loads the env variables from .env
dotenv.load_dotenv('.env', override=True)
# override=True means that if there r variables w the same names in both the .env file
# and also the base env that we r in, the .env file vars will override the ones in base env

# these r retrieving and setting env variables
rabbitmq_host = os.getenv('rabbitmq_host')  # gets hostname or IP
# gets port name that RabbitMQ is listening to
rabbitmq_port = int(os.getenv('rabbitmq_port'))
# gets username for RabbitMQ authentification
rabbitmq_username = os.getenv('rabbitmq_username')
rabbitmq_password = os.getenv('rabbitmq_password')  # gets password for ^
# gets the virtual host within RabbitMQ that the client will connect to
virtual_host = os.getenv('virtual_host')
# the file path where log files will be written
log_file_path = os.getenv('log_file_path')
print("rabbitmq_host", rabbitmq_host, "rabbitmq_port", rabbitmq_port, "rabbitmq_username", rabbitmq_username,
      "rabbitmq_password", rabbitmq_password, "virtual_host", virtual_host, "log_file_path", log_file_path)
# ^ prints out the values of all the retrieved environment variables

# queue and routing key info
# the name of the queue within the RabbitMQ server where messages will be sent for processing
queue_name = 'que_platform_ui_editor_algorithm'
# this is used in message routing to determine which queues should receive the message
routing_key = 'ui_editor_algorithm'
# this s responsible for receiving messages from producers and routing them to queues
defaultmq_sent_exchange_name = 'que_platform_ui_editor_algorithm_result'
# this is the routing key used when sending messages to the exchange
defaultmq_sent_routing_key = 'ui_editor_algorithm_result'

# this initializes the one queue we finna use that passes msgs btw threads
message_queue = Queue()


def mqListen():  # this method listens to Rabbit MQ and processes received msgs

    # we declare these global cuz they hv to be used by multiple methods
    global channel, connection
    # we first hv to make the connections to the server
    connection = pika.BlockingConnection(pika.ConnectionParameters(  # this makes connection
        host=rabbitmq_host, port=rabbitmq_port, virtual_host=virtual_host,  # using pika library
        credentials=pika.PlainCredentials(rabbitmq_username, rabbitmq_password)))  # thru the params we got
    # we use the est. connection to make a channel object
    channel = connection.channel()
    # ^ channel objects r used to talk w server in RabbitMQ
    # we make a queue on the channel called queue_name
    channel.queue_declare(queue=queue_name, durable=True)
    # setting durable=True means that even if the channel restarts, the queue servives
    # this means that the queue that we just made which is acting
    channel.basic_qos(prefetch_count=1)
    # as the collection for the client connection we hv will only fetch 1 new message at a time
    # if we had multiple connections and there was no prefetch_count then one connection might take in
    # a lot of messages at once while the others didn't take any so this ensures that each connection
    # gets one at a time
    # ALSO AND MORE IMPORTANTLY IN THIS CONTEXT since we only hv one client, it means that:
    # we will not get any more messages until the message that we hv rn is finished processing
    # so I'm not 100% sure if this is a good thing bc this means workers cannot process diff msgs concurrently
    # HOWEVER if one msg requires some I/O-bound or CPU-bound tasks, then these tasks can be done concurrently
    # so that means that having multiple workers ISN'T useless if we hv them doing diff tasks for the same msg
    # but if we set prefetch_count to 2 maybeeee that would be more effective?

    # the following is a callback method for processing msgs
    # callback means that it will be passed into another func as an arg for event-driven programming
    # event-driven means like servers and shit

    # the params are as follows (altho properties isn't used)
    def callback(ch, method, properties, body):
        # @ch the channel communication object
        # @method info abt msg delivery
        # @properties the properties of the msg
        # @body the actual msg

        if body:  # so if there's actually a body to the msg
            try:
                # we convert the JSON string msg to Python 'data'
                data = json.loads(body)
                # this is for debugging like as in
                print("+++++++++++++++++++++++++++++++++++++++++")
                # ^ did we acc get to this point if so then we print this
                # we put the msg we parsed into the queue we made earlier
                message_queue.put(data)
                # this part is NECESSARY
                ch.basic_ack(delivery_tag=method.delivery_tag)
                # we r telling RabbitMQ that we acc received the msg
                # so RabbitMQ won't send the msg to anyone else thinking this one didn't receive it
                # now RabbitMQ can remove this msg from their "to-send" queue -is-> queue_name in our code
                # basic_ack - pika method for acknowledging

            # basically if we didn't receive anything (DUH)
            except Exception as e:
                print(f"error: {e}")

    # OK now that we hv the callback method defined this following step is important too
    # we r setting up a consumer to consume msgs from queue_name
    # we use the callback method we j defined to say *this process* is how ur gonna consume msgs from queue_name
    # we hv auto_ack=False bc in our method we hv set up our own acknowledgement just for security
    channel.basic_consume(
        queue=queue_name, on_message_callback=callback, auto_ack=False)
    # this means that we hv started to consume reliably
    print("Starting to consume from RabbitMQ.")
    # ^ this basically shows us like oh everything up til setting up the connection is all g
    # now we just need to start receiving the msgs (which we do using callback)
    channel.start_consuming()  # so let's start


# there is acc no explanation necessary this is like a convoluted switch-case
def process_message(data):
    # for diff msg types that we receive we do diff tasks like yk this
    try:
        print("Processing message:", data)
        message_type = data.get('type')
        if message_type == 1:
            print("*****222logo_position22*****")
            getLogoPosition1(data)
            print("****333logo_position33*****")
        elif message_type == 2:
            print("*****222title_position22*****")
            getTitlePosition1(data)
            print("****333title_position33*****")
        elif message_type == 3:
            print("*****222logo_color22*****")
            getLogoColor1(data)
            print("****333logo_color33*****")
        elif message_type == 4:
            print("*****22222*****")
            getTitlecolor1(data)
            print("****33333*****")
        elif message_type == 6:
            print("*****22222*****")
            getHighlight1(data)
            print("****33333*****")
        elif message_type == 7:
            print("*****22222*****")
            getTitlefont(data)
            print("****33333*****")
        elif message_type == 8:
            print("*****22222*****")
            getTitleLogoPosition(data)
            print("****33333*****")
    except Exception as e:
        print(f"error: {e}")


def worker():  # these workers do the threading part and they r the ones who consume msgs
    while 1:  # this is an infinite loop cuz we never really wanna stop
        try:
            # so RabbitMQ sent us msgs from queue_name
            message = message_queue.get(timeout=1)
            # ^ that we then put into message_queue and now the workers r getting them from
            # ^ message_queue so that they can acc process them
            # note that timeout=1 is set bc that's the max amt of time the get() method waits
            # this is p important here bc without this the worker may wait indef when there's an empty queue
            # which, why is that an issue? prolly bc that blocks other workers
            # this means that we can allow workers to periodically check before moving on
            # ALSO KNOW THAT get() removes the task from the queue

            # we r calling the method to figure out what the msg entails
            process_message(message)
            # and then obvi in this method we acc do the stuff the msg wants us to do asw

            # we basically say to the message_queue that we j did this task
            message_queue.task_done()
            # SO YOU MAY NOT REALIZE BUT these type of queues have an internal counter for tasks
            # and like ya, get() removes the task from the queue but the internal counter of tasks
            # is not decremented yet... BUT NOW IT IS by marking the task as done

        except Exception as e:  # this basically is like if there's an exception j move on
            continue


def listen_data():  # this part acc initiates the threads by using the workers

    threads = []  # this is an empty list of threads
    # we set the max num of workers to 3 so 3 can work concurrently at once
    num_worker_threads = 3

    # this loop j like populates the empty list w the workers
    for _ in range(num_worker_threads):
        thread = threading.Thread(target=worker)  # this makes a new thread
        # ^ the target=worker is specifying which function actually executes the thread - in this case, worker
        thread.start()  # we start the thread
        threads.append(thread)  # we add the thread we j made to the list

    # now we make a single thread for acc listening to RabbitMQ
    listener_thread = threading.Thread(target=mqListen)
    # ^ this uses the mqListen method we made in the first half to do the thread execution
    listener_thread.start()  # we start the thread

    # OK to preface the next two calls, this is the function of join():
    # join is used to wait for a thread to complete its execution before proceeding w the rest of the code
    # when join() is called on a thread, the thread that calls it will pause its execution
    # ^ until the thread that is called is finished executing
    # so like if, in main_thread we call sub_thread.join(), then main_thread will pause until sub_thread is done

    # so below, this means that our main thread (the main program) will:
    # not end until the listener thread completes its task to consume msgs from RabbitMQ
    listener_thread.join()
    message_queue.join()  # and wait for message_queue to be empty before stopping

    # ^ so the above basically says until the msgs are done being consumed from RabbitMQ AND
    # until the workers hv finished processing all msgs, the main thread will not terminate
    # I mean obviously like u can terminate the program manually but this is moreso for:
    # synchronizing thread terminization and stopping pre-mature exits (BOO data leaks)


if __name__ == "__main__":  # this means that this is only ran when it itself is ran directly, not thru any imports or shit
    listen_data()  # now we call this last method to be ran
