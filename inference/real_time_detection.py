import cv2
import numpy as np
from train_model.cnn_lstm_model import predict_on_sequence

IMG_SIZE = 128
SEQUENCE_LENGTH = 16

def real_time_detection():
    frame_buffer = []
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        normalized = resized / 255.0
        frame_buffer.append(normalized)

        if len(frame_buffer) > SEQUENCE_LENGTH:
            frame_buffer.pop(0)

        if len(frame_buffer) == SEQUENCE_LENGTH:
            sequence = np.array(frame_buffer)
            prediction = predict_on_sequence(sequence)

            cv2.putText(frame, f"Prediction: {prediction}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Breathing Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_detection()
