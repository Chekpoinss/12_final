import argparse
import os
import socket
from collections import Counter
from typing import Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from protocol import JsonSocket


EVENT_CLASSES = {"Accident", "Fire", "Flood", "Medical"}
ADDRESS_CLASSES = {"2", "4a", "6", "8a", "10"}

EVENT_TO_TASK = {
    "Fire": "fire",
    "Flood": "flood",
    "Medical": "medical",
    "Accident": "industrial",
}


def infer_house_by_x(x_center: float, frame_width: int) -> str:
    scale = frame_width / 1920.0 if frame_width > 0 else 1.0

    if x_center < 380 * scale:
        return "2"
    elif x_center < 700 * scale:
        return "4a"
    elif x_center < 1050 * scale:
        return "6"
    elif x_center < 1450 * scale:
        return "8a"
    return "10"


def parse_drone_task(
    video_path: str,
    weights_path: str,
    conf: float = 0.25,
    max_frames: int = 180,
    frame_step: int = 2,
) -> Tuple[str, str]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Файл drone-видео не найден: {video_path}")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Файл весов не найден: {weights_path}")

    model = YOLO(weights_path)

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть drone-видео: {video_path}")

    ok, test_frame = cap.read()
    if not ok or test_frame is None:
        cap.release()
        raise RuntimeError(f"Видео открылось, но первый кадр не читается: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    event_votes = Counter()
    address_votes = Counter()
    pair_votes = Counter()

    frame_idx = 0
    used_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        if frame_idx % frame_step != 0:
            continue

        used_frames += 1
        if used_frames > max_frames:
            break

        results = model.predict(source=frame, conf=conf, verbose=False)
        if not results:
            continue

        res = results[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue

        names = getattr(res, "names", model.names)

        xyxy = boxes.xyxy.cpu().numpy()
        clss = boxes.cls.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else [0.0] * len(xyxy)

        h, w = frame.shape[:2]
        frame_area = max(1, w * h)

        best_event = None
        best_event_score = -1.0
        best_event_center_x = None

        best_addr = None
        best_addr_score = -1.0

        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = box.tolist()
            cls_idx = int(clss[i])

            if isinstance(names, dict):
                label = str(names.get(cls_idx, cls_idx))
            else:
                label = str(names[cls_idx])

            score = float(confs[i])
            area = max(1.0, (x2 - x1) * (y2 - y1))
            weighted = score + 2.0 * (area / frame_area)

            if label in EVENT_CLASSES and weighted > best_event_score:
                best_event_score = weighted
                best_event = label
                best_event_center_x = (x1 + x2) / 2.0

            if label in ADDRESS_CLASSES and weighted > best_addr_score:
                best_addr_score = weighted
                best_addr = label

        if best_event is not None and best_event_center_x is not None:
            inferred_addr = infer_house_by_x(best_event_center_x, w)
            event_votes[best_event] += 1
            address_votes[inferred_addr] += 1
            pair_votes[(inferred_addr, best_event)] += 1
        elif best_addr is not None:
            address_votes[best_addr] += 1

        if used_frames % 20 == 0:
            print(
                f"[SERVER][DEBUG] used_frames={used_frames} "
                f"top_pair={pair_votes.most_common(3)} "
                f"top_events={event_votes.most_common(3)} "
                f"top_addr={address_votes.most_common(3)}"
            )

    cap.release()

    if pair_votes:
        (final_addr, final_event), _ = pair_votes.most_common(1)[0]
        print(f"[SERVER] Итог по pair_votes: {pair_votes.most_common(5)}")
        return final_addr, final_event

    if event_votes and address_votes:
        final_event = event_votes.most_common(1)[0][0]
        final_addr = address_votes.most_common(1)[0][0]
        print(f"[SERVER] Итог по event_votes: {event_votes.most_common(5)}")
        print(f"[SERVER] Итог по address_votes: {address_votes.most_common(5)}")
        return final_addr, final_event

    raise RuntimeError(
        "Не удалось определить адрес и тип ЧС по drone-видео.\n"
        f"event_votes={event_votes.most_common(5)}\n"
        f"address_votes={address_votes.most_common(5)}\n"
        f"pair_votes={pair_votes.most_common(5)}"
    )


def detect_candidate_blobs(frame_prev: np.ndarray, frame_cur: np.ndarray):
    h, w = frame_cur.shape[:2]

    y1 = int(h * 0.50)
    prev_roi = frame_prev[y1:, :]
    cur_roi = frame_cur[y1:, :]

    prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(cur_roi, cv2.COLOR_BGR2GRAY)

    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    cur_gray = cv2.GaussianBlur(cur_gray, (5, 5), 0)

    diff = cv2.absdiff(cur_gray, prev_gray)
    _, th = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    th = cv2.morphologyEx(th, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roi_area = max(1, th.shape[0] * th.shape[1])
    roi_h = th.shape[0]

    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 120:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        if bw < 8 or bh < 8:
            continue

        bottom = y + bh
        bottom_ratio = bottom / max(1, roi_h)

        if bottom_ratio < 0.45:
            continue

        blob_ratio = area / roi_area
        center_x = x + bw / 2.0

        candidates.append(
            {
                "x": x,
                "y": y + y1,
                "w": bw,
                "h": bh,
                "center_x": center_x,
                "bottom_ratio": bottom_ratio,
                "blob_ratio": blob_ratio,
            }
        )

    return candidates


def wait_robot_arrival_in_drone_video(video_path: str, address: str) -> bool:
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть drone-видео для детекта прибытия: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 500

    # Ищем только в последней трети ролика
    start_frame = int(total_frames * 0.66)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ok, prev_frame = cap.read()
    if not ok or prev_frame is None:
        cap.release()
        raise RuntimeError("Не удалось прочитать первый кадр для детекта прибытия")

    frame_idx = start_frame + 1
    total_target_hits = 0

    while True:
        ok, cur_frame = cap.read()
        if not ok:
            break

        frame_idx += 1

        candidates = detect_candidate_blobs(prev_frame, cur_frame)
        prev_frame = cur_frame

        target_found_this_frame = False

        for cand in candidates:
            center_x = cand["center_x"]
            blob_ratio = cand["blob_ratio"]
            detected_house = infer_house_by_x(center_x, cur_frame.shape[1])

            # Ослабленный фильтр для дома 2:
            # пропускаем всё, что реально ещё попадает в его сектор
            if address == "2" and center_x > cur_frame.shape[1] * 0.23:
                continue

            if address == "10" and center_x < cur_frame.shape[1] * 0.72:
                continue

            if detected_house != address:
                continue

            if blob_ratio < 0.00012:
                continue

            target_found_this_frame = True
            break

        if target_found_this_frame:
            total_target_hits += 1

        if frame_idx % 10 == 0:
            short = []
            for cand in candidates[:6]:
                short.append(
                    (
                        infer_house_by_x(cand["center_x"], cur_frame.shape[1]),
                        round(cand["center_x"], 1),
                        round(cand["blob_ratio"], 4),
                    )
                )

            print(
                f"[SERVER][ARRIVAL] frame={frame_idx}/{total_frames} "
                f"target={address} candidates={short} "
                f"target_found={target_found_this_frame} "
                f"total_hits={total_target_hits}"
            )

        # Для этого видео этого должно хватать:
        # несколько подтверждений в поздней части = прибытие
        if total_target_hits >= 2:
            cap.release()
            print(f"[SERVER] Детектировано прибытие робота по адресу {address}")
            return True

    cap.release()
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--drone-video", required=True)
    parser.add_argument("--drone-weights", required=True)
    args = parser.parse_args()

    print("[SERVER] Разбираю drone-видео...")
    address, emergency_raw = parse_drone_task(
        video_path=args.drone_video,
        weights_path=args.drone_weights,
        conf=0.25,
        max_frames=180,
        frame_step=2,
    )
    emergency = EVENT_TO_TASK[emergency_raw]

    print(f"[SERVER] Определено: address={address}, emergency={emergency}")

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((args.host, args.port))
    server_sock.listen(1)

    print(f"[SERVER] Жду Robot на {args.host}:{args.port} ...")
    conn, addr = server_sock.accept()
    print(f"[SERVER] Robot подключился: {addr}")

    js = JsonSocket(conn)

    task = {
        "type": "TASK",
        "task_id": "task_001",
        "address": address,
        "emergency": emergency,
    }
    js.send(task)
    print(f"[SERVER] TASK отправлен: {task}")

    try:
        while True:
            msg = js.recv()
            print(f"[SERVER] Получено: {msg}")

            msg_type = msg.get("type")

            if msg_type == "ACK_TASK":
                print("[SERVER] Robot принял задачу")

            elif msg_type == "LOAD_COMPLETE":
                print("[SERVER] Robot сообщил о загрузке")

            elif msg_type == "TRAFFIC_LIGHT_PASSED":
                print("[SERVER] Robot сообщил о проезде светофора")
                print("[SERVER] Жду прибытие робота по drone-видео...")

                arrived = wait_robot_arrival_in_drone_video(
                    video_path=args.drone_video,
                    address=address,
                )

                if arrived:
                    js.send({"type": "STOP", "task_id": "task_001"})
                    print("[SERVER] Отправил STOP роботу")
                else:
                    print("[SERVER] Не удалось уверенно детектировать прибытие робота")
                    print("[SERVER] Отправляю STOP как fallback, чтобы не зависнуть")
                    js.send({"type": "STOP", "task_id": "task_001"})

            elif msg_type == "ACK_STOP":
                print("[SERVER] Robot завершил программу")
                break

    finally:
        js.close()
        server_sock.close()


if __name__ == "__main__":
    main()