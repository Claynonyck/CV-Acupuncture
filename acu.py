import cv2
import mediapipe as mp
import numpy as np

# init
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# cam size
cam_w, cam_h = 640, 480
# panel size
panel_w = 300 

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    # cam size + flip
    image = cv2.resize(cv2.flip(image, 1), (cam_w, cam_h)) 

    # panel ui
    bg = np.zeros((cam_h, cam_w + panel_w, 3), dtype=np.uint8)
    
    # pos
    bg[0:cam_h, 0:cam_w] = image 
    
    cv2.rectangle(bg, (cam_w, 0), (cam_w + panel_w, cam_h), (40, 40, 40), -1)

    cv2.putText(bg, "ACU-RH", (cam_w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.line(bg, (cam_w + 20, 55), (cam_w + panel_w - 20, 55), (100, 100, 100), 2)

    #processing
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            
            # right hand only
            if results.multi_handedness[i].classification[0].label == "Right":
                
                # send to ui
                cv2.putText(bg, "Status: Right Hand Detected", (cam_w + 20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # add landmarks
                mp_drawing.draw_landmarks(
                    bg[0:cam_h, 0:cam_w],
                    hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # acupuncture points add
                #thap tuyen (Shixuan)
                tips = [4, 8, 12, 16, 20]
                for tip in tips:
                    cx = int(hand_landmarks.landmark[tip].x * cam_w)
                    cy = int(hand_landmarks.landmark[tip].y * cam_h)
                    cv2.circle(bg, (cx, cy), 7, (0, 255, 0), -1) 
                
                #hop coc (LI4)
                idx2 = hand_landmarks.landmark[2]
                idx5 = hand_landmarks.landmark[5]
                hc_x = int(((idx2.x + idx5.x) / 2) * cam_w)
                hc_y = int(((idx2.y + idx5.y) / 2) * cam_h)
                cv2.circle(bg, (hc_x, hc_y), 9, (0, 0, 255), -1)
                cv2.putText(bg, "LI4", (hc_x + 15, hc_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                #lao cung (PC8)
                idx9 = hand_landmarks.landmark[9]
                idx0 = hand_landmarks.landmark[0]
                lc_x = int(((idx9.x + idx0.x) / 2) * cam_w)
                lc_y = int(((idx9.y + idx0.y) / 2) * cam_h)
                cv2.circle(bg, (lc_x, lc_y), 9, (255, 0, 0), -1)
                cv2.putText(bg, "PC8", (lc_x + 15, lc_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


                cv2.putText(bg, "Points:", (cam_w + 20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(bg, "LI4", (cam_w + 20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(bg, "PC8", (cam_w + 20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(bg, "Shixuan", (cam_w + 20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.putText(bg, "Live Tracking:", (cam_w + 20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(bg, f"PC8 Pos: (X:{lc_x}, Y:{lc_y})", (cam_w + 20, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Acupuncture', bg)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()