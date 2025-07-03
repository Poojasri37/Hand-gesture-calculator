import cv2
import mediapipe as mp
import numpy as np

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Button size and spacing
button_size = 100
gap = 40

# Spacious Button Layout
buttons = [
    {'label': '1', 'pos': (50, 100)},
    {'label': '2', 'pos': (50 + (button_size + gap), 100)},
    {'label': '3', 'pos': (50 + 2*(button_size + gap), 100)},
    {'label': '+', 'pos': (50 + 3*(button_size + gap), 100)},

    {'label': '4', 'pos': (50, 100 + (button_size + gap))},
    {'label': '5', 'pos': (50 + (button_size + gap), 100 + (button_size + gap))},
    {'label': '6', 'pos': (50 + 2*(button_size + gap), 100 + (button_size + gap))},
    {'label': '-', 'pos': (50 + 3*(button_size + gap), 100 + (button_size + gap))},

    {'label': '7', 'pos': (50, 100 + 2*(button_size + gap))},
    {'label': '8', 'pos': (50 + (button_size + gap), 100 + 2*(button_size + gap))},
    {'label': '9', 'pos': (50 + 2*(button_size + gap), 100 + 2*(button_size + gap))},
    {'label': '*', 'pos': (50 + 3*(button_size + gap), 100 + 2*(button_size + gap))},

    {'label': 'C', 'pos': (50, 100 + 3*(button_size + gap))},
    {'label': '0', 'pos': (50 + (button_size + gap), 100 + 3*(button_size + gap))},
    {'label': '←', 'pos': (50 + 2*(button_size + gap), 100 + 3*(button_size + gap))},
    {'label': '/', 'pos': (50 + 3*(button_size + gap), 100 + 3*(button_size + gap))},
]

# Function to draw buttons
def draw_buttons(image):
    for btn in buttons:
        x, y = btn['pos']
        cv2.rectangle(image, (x, y), (x + button_size, y + button_size), (255, 255, 255), -1)
        cv2.putText(image, btn['label'], (x + 30, y + 65), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)

# State
expression = ""
last_button = None
tap_state = False

# Capture from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    index_finger_tip = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_tip = (int(index_tip.x * w), int(index_tip.y * h))
            cv2.circle(frame, index_finger_tip, 10, (0, 255, 0), -1)

    # Draw calculator display
    draw_buttons(frame)
    cv2.rectangle(frame, (50, 20), (50 + 4*(button_size + gap) - gap, 80), (0, 0, 0), -1)
    cv2.putText(frame, expression, (60, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    # Detect tap over a button (once only)
    if index_finger_tip:
        tapped = False
        for btn in buttons:
            x, y = btn['pos']
            if x < index_finger_tip[0] < x + button_size and y < index_finger_tip[1] < y + button_size:
                if last_button != btn['label'] and not tap_state:
                    tapped = True
                    tap_state = True
                    last_button = btn['label']
                    label = btn['label']

                    if label == 'C':
                        expression = ""
                    elif label == '←':  # Backspace
                        expression = expression[:-1]
                    else:
                        expression += label
                break
        else:
            tap_state = False
            last_button = None

    # Show frame
    cv2.imshow("Hand Gesture Calculator", frame)

    # Keyboard interaction
    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break
    elif key == 13:  # Enter to evaluate
        try:
            expression = str(eval(expression))
        except:
            expression = "Error"

# Cleanup
cap.release()
cv2.destroyAllWindows()
