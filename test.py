import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#for calculating angles
def calculate_angle(a, b, c):
    a = np.array(a) #first
    b = np.array(b) #mid
    c = np.array(c) #end

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

#video feed
pose = mp_pose.Pose()

video_path = "/Users/dreamworld/Desktop/gym_tracker/Bicep Curl Squat Combo.mp4"
cap = cv2.VideoCapture(video_path)

squat_count = 0
curl_count = 0
abs_count = 0

squat_stage = None
curl_stage = None
abs_stage = None

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        #recolour the image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        #make detection
        results = pose.process(image)
        #recolour the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                #extract landmarks
        try:
            landmarks=results.pose_landmarks.landmark

            if results.pose_landmarks:
                # Get landmark coordinates
                landmarks = results.pose_landmarks.landmark

                # Squats - Use right hip, knee, and ankle
                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                squat_angle = calculate_angle(hip, knee, ankle)

                if squat_angle < 90:
                    squat_stage = "down"
                if squat_angle > 160 and squat_stage == "down":
                    squat_stage = "up"
                    squat_count += 1

                # Bicep - Use right shoulder, elbow, and wrist
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                curl_angle = calculate_angle(shoulder, elbow, wrist)

                if curl_angle > 150:
                    curl_stage = "down"
                if curl_angle < 80 and curl_stage == "down":
                    curl_stage = "up"
                    curl_count += 1

                # abs - Use left shoulder, hip, and knee
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                abs_angle = calculate_angle(shoulder, hip, knee)

                if abs_angle > 130:
                    abs_stage = "down"
                if abs_angle < 130 and abs_stage == "down":
                    abs_stage = "up"
                    abs_count += 1


        except:
            pass


        # Display counts
        cv2.putText(image, f'Squats: {squat_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(image, f'Curls: {curl_count}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(image, f'abs: {abs_count}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)


        #render detection
        last_landmarks = None
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245,177,66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                      )

        cv2.imshow("Gym Tracker", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()