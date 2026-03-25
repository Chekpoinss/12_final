import argparse
import socket
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from protocol import JsonSocket


def draw_big_text(
    frame: np.ndarray,
    text: str,
    color: Tuple[int, int, int],
    y_offset: int = 0,
) -> None:
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(1.2, w / 900.0)
    thickness = max(2, int(w / 300))

    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (w - tw) // 2
    y = (h // 2) + y_offset

    cv2.putText(frame, text, (x + 3, y + 3), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def get_label_name(names, cls_idx: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_idx, cls_idx))
    return str(names[cls_idx])


def normalize_label(label: str) -> str:
    return str(label).strip().lower().replace(" ", "").replace("_", "")


def is_target_label(label: str, target_label: str) -> bool:
    return normalize_label(label) == normalize_label(target_label)


def detect_all_warehouses(
    frame: np.ndarray,
    model: YOLO,
    conf: float = 0.20,
) -> List[Dict]:
    results = model.predict(source=frame, conf=conf, verbose=False)
    if not results:
        return []

    res = results[0]
    boxes = getattr(res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    names = getattr(res, "names", model.names)
    xyxy = boxes.xyxy.cpu().numpy()
    clss = boxes.cls.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else [0.0] * len(xyxy)

    h, w = frame.shape[:2]
    frame_area = max(1, h * w)

    detections = []

    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = map(int, box.tolist())
        cls_idx = int(clss[i])
        label = get_label_name(names, cls_idx)
        score = float(confs[i])
        area = max(1.0, float((x2 - x1) * (y2 - y1)))
        area_ratio = area / frame_area

        detections.append(
            {
                "label": label,
                "conf": score,
                "bbox": (x1, y1, x2, y2),
                "area_ratio": area_ratio,
            }
        )

    return detections


def choose_target_detection(
    detections: List[Dict],
    target_label: str,
) -> Optional[Dict]:
    best = None
    best_score = -1.0

    for det in detections:
        if not is_target_label(det["label"], target_label):
            continue

        score = det["conf"] + det["area_ratio"] * 2.0
        if score > best_score:
            best_score = score
            best = det

    return best


def detect_traffic_light_state(frame: np.ndarray) -> Tuple[Optional[str], float, float]:
    h, w = frame.shape[:2]
    roi = frame[0:int(h * 0.40), :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 80, 80], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)

    lower_red2 = np.array([170, 80, 80], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    lower_green = np.array([35, 60, 60], dtype=np.uint8)
    upper_green = np.array([90, 255, 255], dtype=np.uint8)

    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel)

    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_DILATE, kernel)

    total = max(1, roi.shape[0] * roi.shape[1])
    red_ratio = float(np.count_nonzero(red_mask)) / total
    green_ratio = float(np.count_nonzero(green_mask)) / total

    if red_ratio > 0.0015 and red_ratio > green_ratio * 1.15:
        return "red", red_ratio, green_ratio

    if green_ratio > 0.0015 and green_ratio > red_ratio * 1.15:
        return "green", red_ratio, green_ratio

    return None, red_ratio, green_ratio


def detect_stop_line(frame: np.ndarray) -> Tuple[bool, float]:
    h, w = frame.shape[:2]
    roi = frame[int(h * 0.55):, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([15, 70, 70], dtype=np.uint8)
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    yellow_ratio = float(np.count_nonzero(mask)) / max(1, roi.shape[0] * roi.shape[1])

    return yellow_ratio > 0.004, yellow_ratio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-ip", required=True)
    parser.add_argument("--server-port", type=int, default=5000)
    parser.add_argument("--comm-video", required=True)
    parser.add_argument("--warehouse-weights", required=True)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output", default="")
    parser.add_argument("--warehouse-conf", type=float, default=0.20)
    parser.add_argument("--warehouse-stop-area", type=float, default=0.03)
    parser.add_argument("--warehouse-stop-streak", type=int, default=3)
    parser.add_argument("--traffic-green-streak", type=int, default=3)
    parser.add_argument("--traffic-cooldown-seconds", type=float, default=0.2)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.server_ip, args.server_port))
    js = JsonSocket(sock)

    print(f"[ROBOT] Подключился к серверу {args.server_ip}:{args.server_port}")

    msg = js.recv()
    print(f"[ROBOT] Получено: {msg}")

    if msg.get("type") != "TASK":
        raise RuntimeError(f"Ожидал TASK, получил: {msg}")

    emergency = msg["emergency"]
    target_label = emergency
    task_id = msg["task_id"]

    print(
        f"[ROBOT] Принял задачу: address={msg['address']}, "
        f"emergency={emergency}, target_label={target_label}"
    )

    js.send({"type": "ACK_TASK", "task_id": task_id})

    warehouse_model = YOLO(args.warehouse_weights)
    print(f"[ROBOT] model.names = {warehouse_model.names}")

    cap = cv2.VideoCapture(args.comm_video, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.comm_video)

    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть Communication-видео: {args.comm_video}")

    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        cap.release()
        raise RuntimeError(f"Видео открылось, но кадры не читаются: {args.comm_video}")

    h, w = first_frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 25.0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    warehouse_stop_counter = 0
    overlay_stop_frames = 0
    overlay_go_frames = 0

    load_sent = False
    passed_sent = False

    traffic_stage_active = False
    traffic_cooldown_frames = 0

    line_seen_streak = 0
    green_streak = 0
    go_seen_once = False

    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            display = frame.copy()

            # -------- СТАДИЯ 1: поиск нужного склада --------
            if not load_sent:
                detections = detect_all_warehouses(
                    frame=frame,
                    model=warehouse_model,
                    conf=args.warehouse_conf,
                )

                for det in detections:
                    x1, y1, x2, y2 = det["bbox"]
                    label = det["label"]
                    score = det["conf"]
                    area_ratio = det["area_ratio"]

                    color = (255, 180, 0)
                    if is_target_label(label, target_label):
                        color = (0, 255, 255)

                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        display,
                        f"{label} conf={score:.2f} area={area_ratio:.3f}",
                        (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                target_det = choose_target_detection(detections, target_label)

                if frame_idx % 15 == 0:
                    short = [
                        (d["label"], round(d["conf"], 2), round(d["area_ratio"], 3))
                        for d in detections[:8]
                    ]
                    print(
                        f"[ROBOT][DEBUG] frame={frame_idx} target_label={target_label} "
                        f"detections={short}"
                    )

                if target_det is not None:
                    area_ratio = target_det["area_ratio"]
                    if area_ratio >= args.warehouse_stop_area:
                        warehouse_stop_counter += 1
                    else:
                        warehouse_stop_counter = max(0, warehouse_stop_counter - 1)
                else:
                    warehouse_stop_counter = max(0, warehouse_stop_counter - 1)

                cv2.putText(
                    display,
                    f"TARGET_LABEL={target_label} stop_counter={warehouse_stop_counter}/{args.warehouse_stop_streak}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if target_det is not None:
                    cv2.putText(
                        display,
                        f"TARGET area={target_det['area_ratio']:.3f} thresh={args.warehouse_stop_area:.3f}",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                if warehouse_stop_counter >= args.warehouse_stop_streak:
                    overlay_stop_frames = int(fps * 1.5)

                    js.send(
                        {
                            "type": "LOAD_COMPLETE",
                            "task_id": task_id,
                            "warehouse_target_label": target_label,
                        }
                    )
                    print("[ROBOT] Отправил LOAD_COMPLETE")

                    load_sent = True
                    traffic_stage_active = False
                    traffic_cooldown_frames = int(fps * args.traffic_cooldown_seconds)

            # -------- переход к стадии светофора --------
            if load_sent and not passed_sent:
                if not traffic_stage_active:
                    if traffic_cooldown_frames > 0:
                        traffic_cooldown_frames -= 1
                        cv2.putText(
                            display,
                            f"TRAFFIC COOLDOWN: {traffic_cooldown_frames}",
                            (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    else:
                        traffic_stage_active = True

            # -------- СТАДИЯ 2: светофор --------
            if load_sent and not passed_sent and traffic_stage_active:
                traffic_state, red_ratio, green_ratio = detect_traffic_light_state(frame)
                near_stop_line, yellow_ratio = detect_stop_line(frame)

                cv2.putText(
                    display,
                    f"traffic={traffic_state} red={red_ratio:.4f} green={green_ratio:.4f} line={yellow_ratio:.4f}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if near_stop_line:
                    line_seen_streak += 1
                else:
                    line_seen_streak = 0
                    green_streak = 0

                at_traffic_zone = line_seen_streak >= 2

                cv2.putText(
                    display,
                    f"traffic_active={traffic_stage_active} line_streak={line_seen_streak} green_streak={green_streak}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if frame_idx % 10 == 0:
                    print(
                        f"[ROBOT][TRAFFIC] frame={frame_idx} "
                        f"state={traffic_state} red={red_ratio:.4f} green={green_ratio:.4f} "
                        f"line={yellow_ratio:.4f} line_streak={line_seen_streak} green_streak={green_streak}"
                    )

                if at_traffic_zone:
                    if traffic_state == "red":
                        overlay_stop_frames = max(overlay_stop_frames, int(fps * 0.25))
                        green_streak = 0

                    elif traffic_state == "green":
                        overlay_go_frames = max(overlay_go_frames, int(fps * 0.25))
                        go_seen_once = True
                        green_streak += 1

                        if green_streak >= args.traffic_green_streak:
                            js.send({"type": "TRAFFIC_LIGHT_PASSED", "task_id": task_id})
                            print("[ROBOT] Отправил TRAFFIC_LIGHT_PASSED")
                            passed_sent = True

                    else:
                        green_streak = 0

            if overlay_stop_frames > 0:
                draw_big_text(display, "STOP", (0, 0, 255), y_offset=0)
                overlay_stop_frames -= 1

            if overlay_go_frames > 0:
                draw_big_text(display, "GO!", (0, 255, 0), y_offset=90)
                overlay_go_frames -= 1

            cv2.putText(
                display,
                f"frame={frame_idx} load_sent={load_sent} passed_sent={passed_sent}",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(display)

            if args.show:
                cv2.imshow("Robot Communication", display)
                delay_ms = max(1, int(1000 / fps))
                key = cv2.waitKey(delay_ms) & 0xFF
                if key == 27:
                    print("[ROBOT] ESC -> выход")
                    break

        # fallback: если GO уже был показан, но ролик закончился чуть раньше сообщения
        if load_sent and not passed_sent and go_seen_once:
            js.send({"type": "TRAFFIC_LIGHT_PASSED", "task_id": task_id})
            print("[ROBOT] Видео кончилось после GO!, отправил TRAFFIC_LIGHT_PASSED как fallback")
            passed_sent = True

        print("[ROBOT] Видео завершено, жду STOP от сервера...")
        while True:
            msg = js.recv()
            print(f"[ROBOT] Получено: {msg}")
            if msg.get("type") == "STOP":
                print("[ROBOT] Получил STOP. Завершаюсь.")
                js.send({"type": "ACK_STOP", "task_id": msg.get("task_id", "")})
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        js.close()


if __name__ == "__main__":
    main()