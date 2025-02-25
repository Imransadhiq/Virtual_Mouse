import cv2
import mediapipe as mp
import pyautogui

# Initialize hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Get screen size
screen_width, screen_height = pyautogui.size()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue
    
    # Flip image and convert to RGB
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process image for hand tracking
    results = hands.process(rgb_image)
    height, width, _ = image.shape
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get index finger tip coordinates
            index_x = int(hand_landmarks.landmark[8].x * width)
            index_y = int(hand_landmarks.landmark[8].y * height)
            
            # Convert to screen coordinates
            screen_x = int((index_x / width) * screen_width)
            screen_y = int((index_y / height) * screen_height)
            
            # Move mouse
            pyautogui.moveTo(screen_x, screen_y)
            
            # Thumb tip for clicking (landmark[4])
            thumb_x = int(hand_landmarks.landmark[4].x * width)
            thumb_y = int(hand_landmarks.landmark[4].y * height)
            
            # Check if thumb and index are close (for clicking)
            if abs(index_x - thumb_x) < 30 and abs(index_y - thumb_y) < 30:
                pyautogui.click()
                cv2.putText(image, 'Click!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Check if all fingers are up (Gesture to open folder)
            finger_up = [hand_landmarks.landmark[i].y < hand_landmarks.landmark[i - 2].y for i in [8, 12, 16, 20]]
            if all(finger_up):
                pyautogui.hotkey('win', 'e')  # Open file explorer
                cv2.putText(image, 'Opening Folder', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Virtual Mouse', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
