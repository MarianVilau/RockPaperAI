import sensor, image, time, os, tf, uos, gc
import random, pyb

history = {"rock": 0, "paper": 0, "scrissor": 0}
redLED = pyb.LED(1)
greenLED = pyb.LED(2)
blueLED = pyb.LED(3)

def init_sensor():
    sensor.reset()
    sensor.set_pixformat(sensor.GRAYSCALE)
    sensor.set_framesize(sensor.QVGA)
    sensor.set_windowing((240, 240))
    sensor.skip_frames(time=2000)

def load_model_and_labels():
    try:
        net = tf.load("trained.tflite", load_to_fb=uos.stat('trained.tflite')[6]
                                                   > (gc.mem_free() - (64*1024)))
    except Exception as e:
        print(e)
        raise Exception('Failed to load "trained.tflite", did you copy the .tflite and ' +
                        'labels.txt file onto the mass-storage device? (' + str(e) + ')')

    try:
        labels = [line.rstrip('\n') for line in open("labels.txt")]
    except Exception as e:
        raise Exception('Failed to load "labels.txt", did you copy the .tflite and ' +
                        'labels.txt file onto the mass-storage device? (' + str(e) + ')')

    return net, labels

def countdown_timer():
    for i in range(3, 0, -1):
        img = sensor.snapshot()
        img.draw_string(10, 10, str(i), color=255, scale=2)
        sensor.flush()
        time.sleep(1)

def process_and_classify_images(net, labels):
    img = sensor.snapshot()
    countdown_timer()

    max_prediction = None
    max_confidence = -1

    for obj in net.classify(img, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
        print("**********\n")
        img.draw_rectangle(obj.rect())
        predictions_list = list(zip(labels, obj.output()))

        for i in range(len(predictions_list)-1):
            if predictions_list[i][1] > max_confidence:
                max_confidence = predictions_list[i][1]
                max_prediction = predictions_list[i][0]

    if max_prediction is not None:
        print("Your move is:" + "\033[1;32;40m ", max_prediction, "\033[0m")
        update_history(max_prediction)

    return max_prediction

def update_history(user_move):
    global history
    history[user_move] += 1


def predict_user_move():
    global history
    total_moves = sum(history.values())
    if total_moves == 0:
        return random.choice(["rock", "paper", "scrissor"])
    probabilities = {move: count / total_moves for move, count in history.items()}
    return max(probabilities, key=probabilities.get)

# Hidden Markov models
#minmax why not
def play_against_software(user_move):
    predicted_move = predict_user_move()
    if predicted_move == "rock":
        software_move = "paper"
    elif predicted_move == "scrissor":
        software_move = "rock"
    else:
        software_move = "scrissor"

    print("Software move is:" + "\033[1;31;40m ", software_move, "\033[0m")

    if user_move == software_move:
        blueLED.on()
        return 0

    elif (user_move == "rock" and software_move == "scrissor") or \
         (user_move == "scrissor" and software_move == "paper") or \
         (user_move == "paper" and software_move == "rock"):
        greenLED.on()
        return 1

    else:
        redLED.on()
        return -1

def print_header():
    print("\n\033[1;36;40m=========================================")
    print("           ROCK PAPER SCISSORS           ")
    print("=========================================\033[0m")

def print_score(score):
    print("\033[1;33;40m\nCurrent score: " + str(score) + "\033[0m")

def print_error():
    print("\033[1;31;40mInvalid option. Please try again.\033[0m")

def main():
    init_sensor()
    net, labels = load_model_and_labels()
    score = 0
    print_header()
    while(True):
        user_move = process_and_classify_images(net, labels)
        time.sleep(1)
        if user_move is not None:
            result = play_against_software(user_move)
            score += result
            print_score(score)
            time.sleep(1)
            user_move = None
        else:
            print_error()
        print("Prepare for next move in: ")
        for i in range (3, 0, -1):
            print(i, end=' ')
            time.sleep(1)
        print("\nShow your next move\n")
        greenLED.off()
        redLED.off()
        blueLED.off()

if __name__ == "__main__":
    main()
