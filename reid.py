import cv2
import numpy as np

# Load pre-trained person detection model (Haar Cascade)
person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Initialize video capture
cap = cv2.VideoCapture('C:/Users/devas/OneDrive/Desktop/miniproject/v4.mp4')

# Dictionary to store detected persons and their IDs
detected_persons = {}

# Counter for the number of people
person_count = 0

# Set the minimum distance for re-identification
min_distance = 50

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale for person detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect persons in the frame
    persons = person_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in persons:
        # Check if this person has been detected before
        for person_id, (prev_x, prev_y, _, _) in detected_persons.items():
            # Calculate Euclidean distance between current and previous person
            distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)

            # Compare distances
            if distance < min_distance:
                break
        else:
            # Assign a new ID to the person
            person_count += 1
            detected_persons[person_count] = (x, y, x+w, y+h)

        # Draw a rectangle around the person
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the person ID
        cv2.putText(frame, f"Person {person_count}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the total number of people
    cv2.putText(frame, f"Total People: {person_count}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Person Re-identification', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
#devashishbisht