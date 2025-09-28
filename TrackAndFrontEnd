import streamlit as st
import cv2
import time

st.title("speechtomizor")


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


if "run" not in st.session_state:
    st.session_state.run = False


start_button = st.button("Start")
stop_button = st.button("Stop")


if start_button:
    st.session_state.run = True
if stop_button:
    st.session_state.run = False


raw_placeholder = st.empty()
timer_placeholder = st.empty()


if st.session_state.run:
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        st.error("Could not open camera.")
    else:
        start_time = None
        total_time = 0

        while st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab frame.")
                break

            raw_frame = frame.copy()
            img = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
            eyes_detected = False

            faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi_gray = img[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray)

                if len(eyes) >= 2:
                    eyes_detected = True
                    for (ex, ey, ew, eh) in eyes[:2]:
                        center = (x + ex + ew // 2, y + ey + eh // 2)
                        radius = int(round((ew + eh) * 0.25))
                        cv2.circle(frame, center, radius, (255, 0, 0), 2)


            if eyes_detected:
                if start_time is None:
                    start_time = time.time()
                elapsed = time.time() - start_time
                total_time += elapsed
                start_time = time.time()
            else:
                start_time = None


            timer_text = f"Eyes on camera: {total_time:.1f} sec"
            cv2.putText(raw_frame, timer_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


            raw_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            raw_placeholder.image(raw_rgb, channels="RGB")


        cap.release()
        cv2.destroyAllWindows()


        timer_placeholder.success(f"Final Eyes on Camera Time: {total_time:.1f} sec")
