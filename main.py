import cv2
import numpy as np
import os

NO_MOTION_DURATION_MS = 500
SHOW_VIDEO = False


def detect_and_create_clips(
    video_path,
    threshold=30,
    contour_area_threshold=1000,
    no_motion_duration=NO_MOTION_DURATION_MS,
):
    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    motion_started = False
    start_time = 0
    out = None  # VideoWriter object for saving clips

    frame_count = 0
    clip_index = 0

    while cap.isOpened():
        frame_count += 1

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
        motion_detected = any(
            cv2.contourArea(c) > contour_area_threshold for c in contours
        )

        if motion_detected and not motion_started:
            # This is the start of motion
            start_time = cap.get(cv2.CAP_PROP_POS_MSEC)
            motion_started = True

            # Create a video writer object here to start writing the clip
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            clip_name = f"output_clips/temp_clip_{clip_index}.mp4"
            out = cv2.VideoWriter(
                clip_name,
                fourcc,
                cap.get(cv2.CAP_PROP_FPS),
                (
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                ),
            )

        elif not motion_detected and motion_started:
            # This is the end of motion
            end_time = cap.get(cv2.CAP_PROP_POS_MSEC)

            if (end_time - start_time) > no_motion_duration:
                # Rename the clip to include its duration
                clip_length = int((end_time - start_time) / 1000)
                final_clip_name = (
                    f"output_clips/clip_{clip_index + 1}_{clip_length}s.mp4"
                )
                if out:
                    out.release()
                    os.rename(
                        f"output_clips/temp_clip_{clip_index}.mp4", final_clip_name
                    )
                    clip_index += 1
                print(f"Saved clip from {start_time}ms to {end_time}ms")
            else:
                # Delete the temporary clip
                if out:
                    out.release()
                    os.remove(f"output_clips/temp_clip_{clip_index}.mp4")
                print(
                    f"Skipped potential clip from {start_time}ms to {end_time}ms due to short duration."
                )
            motion_started = False
            out = None

        # Write frame to the current clip if motion is detected
        if motion_started and out:
            out.write(frame1)

        if SHOW_VIDEO:
            cv2.imshow("Preview", frame1)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame1 = frame2
        ret, frame2 = cap.read()
        if not ret:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "input_videos/test_video.mp4"
    detect_and_create_clips(video_path)
