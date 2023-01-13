import cv2 
import mediapipe as mp 
 
# webcam video capture 
video = cv2.VideoCapture(2) 
 
# configure mediapipe element 
hand = mp.solutions.hands 
 
#using only one hand 
my_hand = hand.Hands(max_num_hands=1) 
 
# draw hand lines 
draw = mp.solutions.drawing_utils 
 
while True: 
    window_name = 'Finger counter'
    
    # receving image from webcam 
    ret, frame = video.read() 
    
    if not ret:
        break
     
    # convert image from brg format (webcam) to rgb  
    imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    
    # processing image    
    result_image = my_hand.process(imageRGB) 
    hand_points = result_image.multi_hand_landmarks
    
    h, w, _ = frame.shape
    finger_points = []
    
    # get coordinates from finger marks
    if hand_points:
        for marks in hand_points:
            # draw.draw_landmarks(frame, marks, hand.HAND_CONNECTIONS)
            for id, coordinate in enumerate(marks.landmark):
                coord_x, coord_y = int(coordinate.x * w), int(coordinate.y * h)
                finger_points.append((coord_x, coord_y))
        
        fingers = [8, 12, 16, 20]
        counter = 0
        
        # count fingers
        if marks : 
            for item in fingers:
                if finger_points[item][1] < finger_points[item -2][1]:
                    counter += 1
            if finger_points[4][0] < finger_points[3][0]:
                counter += 1
        
        # show counter 
        cv2.putText(frame, str(counter), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)
            
    cv2.imshow(window_name, frame) 
    cv2.waitKey(1)    


video.release()
cv2.destroyAllWindows()

