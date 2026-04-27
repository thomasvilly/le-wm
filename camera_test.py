import cv2

# List of camera indexes to try. 
# 0 is usually the default/built-in webcam.
# 1, 2, etc., are usually USB webcams.
camera_indexes = [1] 

# List to hold open camera objects
captures = []

# 1. Initialize all cameras
print("Initializing cameras...")
for index in camera_indexes:
    # Change the line inside the loop to:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    exp_val = -6
    if exp_val != 0:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
        cap.set(cv2.CAP_PROP_EXPOSURE, exp_val)
        print(f"Exposure set to: {exp_val}")

    if cap.isOpened():
        print(f"Camera {index} opened successfully.")
        captures.append(cap)
    else:
        print(f"Warning: Could not open camera {index}.")

if not captures:
    print("No cameras found. Exiting.")
    exit()

print("Press 'q' to quit.")

# 2. Loop to read and display frames
while True:
    for i, cap in enumerate(captures):
        # Read a frame from this camera
        ret, frame = cap.read()
        
        if ret:
            # Resize frame optional: easier to fit multiple on screen
            # frame = cv2.resize(frame, (640, 480))
            
            # Show the frame in a window unique to this camera
            window_name = f"Camera {camera_indexes[i]}"
            cv2.imshow(window_name, frame)
    
    # Check if user pressed 'q' to quit
    # waitKey(1) waits 1ms for a key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 3. Cleanup: release cameras and close windows
for cap in captures:
    cap.release()
cv2.destroyAllWindows()