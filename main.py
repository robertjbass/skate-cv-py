import cv2
import os

NO_MOTION_DURATION_MS = 500
SHOW_VIDEO = False
USER = "bob[at]bbass[dot]co"
INPUT_FOLDER = f"videos/{USER}/input"
OUTPUT_FOLDER = f"videos/{USER}/output"

# TODO - add buffer to start and end of clips
CLIP_START_BUFFER_MS = 1000
CLIP_END_BUFFER_MS = 1000


def display_ms_as_minutes_and_seconds(ms):
    seconds = round(ms / 1000)
    if seconds < 60:
        return f"{seconds}s"
    else:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes}m{seconds}s"


def detect_and_create_clips(
    video_path,
    threshold=30,
    contour_area_threshold=1000,
    no_motion_duration=NO_MOTION_DURATION_MS,
    video_count_string="",
):
    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    # Get video frame count
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    motion_started = False
    start_time = 0
    # VideoWriter object for saving clips
    out = None

    frame_count = 0
    clip_index = 0

    percent_complete = 0
    clips_saved = 0

    files_in_output_folder = os.listdir(OUTPUT_FOLDER)
    existing_file_count = len(files_in_output_folder)

    # Create a video writer object here to start writing the clip
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    capture_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    capture_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("Processing video...")
    while cap.isOpened():
        frame_count += 1

        percent = round(frame_count / total_frame_count * 100)
        if percent > percent_complete:
            percent_complete = percent
            if percent_complete % 5 == 0:
                print(f"Video {video_count_string} | {percent_complete}% complete")

        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)

        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        motion_detected = any(
            cv2.contourArea(c) > contour_area_threshold for c in contours
        )

        clip_number = existing_file_count + clip_index + 1

        temporary_clip_path = f"{OUTPUT_FOLDER}/temp_clip_{clip_number}.mp4"
        complete_clip_path = f"{OUTPUT_FOLDER}/clip_{clip_number}.mp4"

        if motion_detected and not motion_started:
            # This is the start of motion
            start_time = cap.get(cv2.CAP_PROP_POS_MSEC)
            motion_started = True

            out = cv2.VideoWriter(
                temporary_clip_path,
                fourcc,
                fps,
                (capture_width, capture_height),
            )

        elif not motion_detected and motion_started:
            # This is the end of motion
            end_time = cap.get(cv2.CAP_PROP_POS_MSEC)

            if (end_time - start_time) > no_motion_duration:
                # Rename the clip to include its duration
                clip_length = int((end_time - start_time) / 1000)

                if out:
                    out.release()
                    os.rename(temporary_clip_path, complete_clip_path)
                    clip_index += 1
                    clips_saved = clips_saved + 1
                print(
                    f"Saved clip from {display_ms_as_minutes_and_seconds(start_time)} to {display_ms_as_minutes_and_seconds(end_time)}"
                )
            else:
                # Delete the temporary clip
                if out:
                    out.release()
                    os.remove(f"{OUTPUT_FOLDER}/temp_clip_{clip_number}.mp4")
                print(
                    f"Skipped potential clip at {display_ms_as_minutes_and_seconds(end_time)} due to duration under {no_motion_duration}ms"
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
    return clips_saved


def main():
    if not os.path.exists(INPUT_FOLDER):
        print(f"Video input path not found")
        return

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    input_files = os.listdir(INPUT_FOLDER)
    print(f"Found {len(input_files)} files in {INPUT_FOLDER}")

    for index, file in enumerate(input_files):
        video_path = f"{INPUT_FOLDER}/{file}"

        clips_saved_count = 0
        clips_saved = detect_and_create_clips(
            video_path=video_path,
            video_count_string=f"{index + 1} of {len(input_files)}",
        )

        clips_saved_count = clips_saved_count + clips_saved

    print(f"Complete - saved {clips_saved} clips from {len(input_files)} videos")


if __name__ == "__main__":
    main()
