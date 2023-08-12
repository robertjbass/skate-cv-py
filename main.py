import cv2
import numpy as np

NO_MOTION_DURATION_MS = 500

# previewing video is slow, so only do it if you need to
SHOW_VIDEO = True


def detect_movement(
    video_path,
    threshold=30,
    contour_area_threshold=1000,
    no_motion_duration=NO_MOTION_DURATION_MS,
):
    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    clips = []
    motion_started = False
    start_time = 0

    frame_count = 0

    while cap.isOpened():
        frame_count += 1
        print("frame", frame_count)

        # process every other frame
        if frame_count % 2 == 0:
            frame1 = frame2
            ret, frame2 = cap.read()

            continue

        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)

        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False

        for contour in contours:
            if cv2.contourArea(contour) > contour_area_threshold:
                # Draw the contour in green
                cv2.drawContours(frame1, [contour], -1, (0, 255, 0), 2)
                motion_detected = True

        if motion_detected and not motion_started:
            # This is the start of motion
            start_time = cap.get(cv2.CAP_PROP_POS_MSEC)
            motion_started = True
        elif not motion_detected and motion_started:
            # This is the end of motion
            end_time = cap.get(cv2.CAP_PROP_POS_MSEC)

            # Check if time between start and end exceeds the minimum duration
            if (end_time - start_time) > no_motion_duration:
                print("saving clip")
                clips.append({"start": start_time, "end": end_time})
            else:
                print(
                    f"Skipped potential clip from {start_time}ms to {end_time}ms due to short duration."
                )  # Debug line

            motion_started = False

        # Display the processed frame
        if SHOW_VIDEO:
            cv2.imshow("Preview", frame1)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame1 = frame2
        ret, frame2 = cap.read()

        if not ret:
            break

    # If motion was ongoing at the end of the video, capture that too
    if motion_started:
        clips.append({"start": start_time, "end": cap.get(cv2.CAP_PROP_POS_MSEC)})

    cap.release()
    cv2.destroyAllWindows()

    return clips


if __name__ == "__main__":
    video_path = "input_videos/test_video.mp4"
    motion_clips = detect_movement(video_path)

    print(motion_clips)
