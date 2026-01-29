import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time

# --- CONFIGURATION ---
cam_w, cam_h = 640, 480   
frame_margin = 100        
smoothening = 5           
CLICK_THRESHOLD = 40      

# Fix pyautogui lag
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# Init Camera
cap = cv2.VideoCapture(0)
cap.set(3, cam_w)
cap.set(4, cam_h)

# Init Hand Tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, 
                       min_detection_confidence=0.7, 
                       min_tracking_confidence=0.7)

# Screen Size
screen_w, screen_h = pyautogui.size()

# State Variables
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0
last_click_time = 0

def get_fingers_up(hand):
    fingers = []
    # Thumb
    fingers.append(hand.landmark[4].x < hand.landmark[3].x)
    # 4 Fingers
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        fingers.append(hand.landmark[tip].y < hand.landmark[pip].y)
    return fingers

while True:
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Draw Active Region Box
    cv2.rectangle(frame, (frame_margin, frame_margin), 
                  (w - frame_margin, h - frame_margin), (255, 0, 255), 2)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            
            lm = hand.landmark
            x1, y1 = int(lm[8].x * w), int(lm[8].y * h)  # Index Tip
            x2, y2 = int(lm[12].x * w), int(lm[12].y * h) # Middle Tip
            x_thumb, y_thumb = int(lm[4].x * w), int(lm[4].y * h) # Thumb Tip
            
            fingers = get_fingers_up(hand)
            # fingers[1] = Index, fingers[2] = Middle
            
            # --- MODE 1: MOVE CURSOR (Index Up, Middle Down) ---
            if fingers[1] and not fingers[2]:
                
                # Move Logic
                x3 = np.interp(x1, (frame_margin, w - frame_margin), (0, screen_w))
                y3 = np.interp(y1, (frame_margin, h - frame_margin), (0, screen_h))
                
                curr_x = prev_x + (x3 - prev_x) / smoothening
                curr_y = prev_y + (y3 - prev_y) / smoothening
                
                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y
                
                # Visual Line (Index to Thumb)
                dist_click = math.hypot(x1 - x_thumb, y1 - y_thumb)
                
                # Check for Left Click
                if dist_click < CLICK_THRESHOLD:
                    cv2.circle(frame, (x1, y1), 15, (0, 0, 255), cv2.FILLED) # Red
                    cv2.putText(frame, "LEFT CLICK", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if time.time() - last_click_time > 0.3:
                        pyautogui.click()
                        last_click_time = time.time()
                else:
                    cv2.circle(frame, (x1, y1), 15, (0, 255, 0), cv2.FILLED) # Green
                    # Draw Line
                    cv2.line(frame, (x1, y1), (x_thumb, y_thumb), (0, 255, 0), 2)

            # --- MODE 2: RIGHT CLICK (Index & Middle Up) ---
            elif fingers[1] and fingers[2]:
                # Visual Line (Middle to Thumb)
                dist_right = math.hypot(x2 - x_thumb, y2 - y_thumb)
                
                # Check for Right Click
                if dist_right < CLICK_THRESHOLD:
                    cv2.circle(frame, (x2, y2), 15, (0, 0, 255), cv2.FILLED) # Red
                    cv2.putText(frame, "RIGHT CLICK", (x2, y2-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if time.time() - last_click_time > 0.3:
                        pyautogui.rightClick()
                        last_click_time = time.time()
                else:
                    cv2.circle(frame, (x2, y2), 15, (255, 0, 0), cv2.FILLED) # Blue
                    cv2.line(frame, (x2, y2), (x_thumb, y_thumb), (255, 0, 0), 2)

    # --- INSTRUCTIONS PANEL (Black Box at Bottom) ---
    # 1. Draw black background
    cv2.rectangle(frame, (0, h-60), (w, h), (0, 0, 0), -1)
    
    # 2. Add Text Instructions
    cv2.putText(frame, "ONE FINGER UP: Move Mouse", (10, h - 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.putText(frame, "PINCH INDEX+THUMB: Left Click", (10, h - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
    cv2.putText(frame, "TWO FINGERS UP + PINCH MIDDLE: Right Click", (300, h - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imshow("AI Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()