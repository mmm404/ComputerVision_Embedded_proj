import cv2
import mediapipe as mp
import serial
import math

# Initialize mediapipe hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils
video_path = r'C:\Users\pc\Desktop\mechatronics folder\cded\code13\project_dataset\vid1.mp4'
cap = cv2.VideoCapture(video_path)  # video parameter
max_val = 0
def compute_magnitude(i,j,k,l):
    x = i
    y = j
    x1 = k 
    y1 = l
    return math.isqrt(round(math.pow(abs(x-x1),2) + math.pow(abs(y-y1),2)))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for idx, landmark in enumerate(hand_landmarks.landmark):

                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.putText(frame, f'{idx}: ({cx}, {cy})', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                if idx == 4:
                    i = cx
                    j = cy
                if idx == 8:
                    k = cx
                    l = cy
                    flex = compute_magnitude(i,j,k,l)
                    if flex > max_val:
                        max_val = flex
                    print(flex)
                    ser = serial.Serial('COM4', 9600)
    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
print(f"Maximum flex value : {max_val}")
cap.release()
cv2.destroyAllWindows()
