from function import *
from time import sleep

for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except FileExistsError:
            pass

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):

                # Read image
                image_path = 'Image/{}/{}.png'.format(action, sequence)
                frame = cv2.imread(image_path)

                if frame is None:
                    print(f"Error reading image: {image_path}")
                    continue

                # Make detections
                image, results = mediapipe_detection(frame, hands)

                if image is None or results is None:
                    print(f"Error in mediapipe_detection for {image_path}")
                    continue

                # Draw landmarks
                draw_styled_landmarks(image, results)

                # Display information
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                # Show to screen
                cv2.imshow('OpenCV Feed', image)

                # Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()
