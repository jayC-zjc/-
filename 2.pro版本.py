from ultralytics import YOLO
import cv2
import numpy as np

# ================= 模型（路径不变） =================
model = YOLO(r"C:\Users\USER\Desktop\runs\detect\train-3\weights\best.pt")

video_path = r"C:\Users\USER\Desktop\task2\路口素材2.mp4"
save_path = r"C:\Users\USER\Desktop\task2\final_result.mp4"

cap = cv2.VideoCapture(video_path)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    save_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (w, h)
)

# ================= 计数线 =================
LINE_Y = 150

# ================= 参数 =================
DIST_THRESH = 60
MAX_MISS = 15
INIT_FRAMES = 30

# ================= 轨迹池 =================
track_pool = {}

counted_tracks = set()
initial_tracks = set()

flash_frames = {}
FLASH = 20

total_count = 0
frame_id = 0

def dist(x1, y1, x2, y2):
    return np.hypot(x1 - x2, y1 - y2)

# ================= 主循环 =================
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.35,
        iou=0.5
    )

    annotated = frame.copy()

    current_detections = []

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        for box, tid in zip(boxes, ids):

            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            tid = int(tid)

            current_detections.append((cx, cy, tid, box))

    # ================= 🔥【关键修复：初始车辆判断】 =================
    # ❌ 原来：按 frame_id 直接全部屏蔽
    # ✅ 现在：只屏蔽“已经在计数线以上的车”

    for cx, cy, tid, _ in current_detections:

        # 👉 只有“已经明显在上方”的车才算初始车
        if cy < LINE_Y - 40:
            initial_tracks.add(tid)

    matched_pool = set()

    for cx, cy, tid, box in current_detections:

        best_id = None
        best_dist = 999

        for pid, p in track_pool.items():

            d = dist(cx, cy, p["cx"], p["cy"])

            if d < DIST_THRESH and d < best_dist:
                best_dist = d
                best_id = pid

        track_id = best_id if best_id is not None else tid

        matched_pool.add(track_id)

        if track_id not in track_pool:

            track_pool[track_id] = {
                "cx": cx,
                "cy": cy,
                "miss": 0,
                "counted": False,
                "last_y": cy
            }

        else:

            prev_y = track_pool[track_id]["last_y"]

            if prev_y > LINE_Y and cy <= LINE_Y:

                if (not track_pool[track_id]["counted"]
                    and track_id not in initial_tracks):

                    track_pool[track_id]["counted"] = True
                    counted_tracks.add(track_id)
                    total_count += 1

                    flash_frames[track_id] = FLASH

                    print(f"COUNT +1 → ID:{track_id}")

            track_pool[track_id]["cx"] = cx
            track_pool[track_id]["cy"] = cy
            track_pool[track_id]["last_y"] = cy
            track_pool[track_id]["miss"] = 0

    for pid in list(track_pool.keys()):

        if pid not in matched_pool:

            track_pool[pid]["miss"] += 1

            if track_pool[pid]["miss"] > MAX_MISS:
                del track_pool[pid]

    # ================= 可视化 =================
    if results[0].boxes.id is not None:

        for cx, cy, tid, box in current_detections:

            x1, y1, x2, y2 = box

            show_id = None

            for pid, p in track_pool.items():
                if dist(cx, cy, p["cx"], p["cy"]) < DIST_THRESH:
                    show_id = pid
                    break

            if show_id in flash_frames:

                cv2.rectangle(annotated,
                              (int(x1), int(y1)),
                              (int(x2), int(y2)),
                              (0, 0, 255), 4)

                cv2.putText(annotated,
                            "COUNTED",
                            (int(x1), int(y1)-30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255), 3)

                flash_frames[show_id] -= 1
                if flash_frames[show_id] <= 0:
                    del flash_frames[show_id]

            else:

                cv2.rectangle(annotated,
                              (int(x1), int(y1)),
                              (int(x2), int(y2)),
                              (0, 255, 0), 2)

            if show_id is not None:

                cv2.putText(annotated,
                            f"ID:{show_id}",
                            (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0), 2)

            cv2.circle(annotated,
                       (int(cx), int(cy)),
                       3,
                       (0, 0, 255), -1)

    cv2.line(annotated,
             (0, LINE_Y),
             (w, LINE_Y),
             (255, 0, 0), 3)

    cv2.putText(annotated,
                f"COUNT: {total_count}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255), 3)

    out.write(annotated)

    cv2.imshow("Robust Vehicle Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================= 结束 =================
cap.release()
out.release()
cv2.destroyAllWindows()

print("最终车辆数:", total_count)