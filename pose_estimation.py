from ultralytics import YOLO
import cv2
import numpy as np
import math
import time
import csv

model = YOLO("yolo11s-pose.pt")
cap = cv2.VideoCapture('VIDEO_NAME.mp4') 

# Video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  
out = cv2.VideoWriter("processed_video_name.mp4", fourcc, fps, (width, height))

person_colors = {
    0:(0,255,0),
    1:(0,0,255),
    2:(255,0,0),
    3:(0,255,255),
    4:(255,255,0),
    5:(127,127,0),
    6:(255,127,0),
    7:(128,0,255) } # Kişi ID'leri için renkler

# eklem bağlantı eşleşmeleri
SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16)
]

frame_number = 0
center_points_cur_frame = []
center_points_prev_frame = []

# tracker alg. degiskenleri
tracking_objects = {}
tracking_keypoints = {}
tracking_boxes = {}
tracking_pose_status = {} 
tracking_velocities = {}  #
track_id = 0

person_visible_kpts = {} # for validationing keypoints
pose_log_records = [] # for csv

# running durum degiskenleri
running_history_size = 5  # Kaç kare boyunca hız izlenecek
running_counter_threshold = 2  # Kaç kare koşma tespit edilirse running kabul edilecek
running_counters = {}  # Her kişi için koşma sayacı

def calculate_angle(a, b, c): # 3 nokta ile eklem açısı hesaplar
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def analyze_pose(keypoints, bbox, object_id, squat_angle_threshold=130, running_speed_threshold = 10): # koşma, oturma, yatma, ayakta durma eylemlerini tespit eder
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    ratio = w / h

    # Koşma durumunu kontrol eder
    if object_id in tracking_velocities:
        recent_velocities = tracking_velocities[object_id]
        avg_velocity = sum(recent_velocities) / len(recent_velocities) if recent_velocities else 0
        
        if avg_velocity > running_speed_threshold:
            if object_id not in running_counters:
                running_counters[object_id] = 0
            running_counters[object_id] += 1
        else:
            if object_id in running_counters:
                running_counters[object_id] = max(0, running_counters[object_id] - 1)
        
        if object_id in running_counters and running_counters[object_id] >= running_counter_threshold:
            return "running"

    # ayakta olma veya yatma durumunu kontrol eder
    if ratio > 1.5:
        return "lying"
    elif ratio < 0.5:
        return "standing"
    
    try:
        # Gerekli noktaların varlığını sorgular eksik varsa none döner
        needed = [9,10,11, 12, 13, 14, 15, 16]  # hips, knees, ankles
        for i in needed:
            if i >= len(keypoints) or keypoints[i][0] <= 0 or keypoints[i][1] <= 0:
                return None

        # Sol bacak açısı
        l_hip = keypoints[11][:2]
        l_knee = keypoints[13][:2]
        l_ankle = keypoints[15][:2]
        left_angle = calculate_angle(l_hip, l_knee, l_ankle)

        # Sağ bacak açısı
        r_hip = keypoints[12][:2]
        r_knee = keypoints[14][:2]
        r_ankle = keypoints[16][:2]
        right_angle = calculate_angle(r_hip, r_knee, r_ankle)

        # Ortalama diz açısı
        avg_angle = (left_angle + right_angle) / 2
        
        # el bileği - ayak bileği mesafesi analizi (Y ekseni)  / Bu sayede dizlerini kırmadan eğilen birini tespit edebiliriz
        l_wrist_y = keypoints[9][1]
        r_wrist_y = keypoints[10][1]
        ankle_ys = [keypoints[15][1], keypoints[16][1]]

        threshold = h * 0.3
        wrist_near_ankle = False

        for wrist_y in [l_wrist_y, r_wrist_y]:
            for ankle_y in ankle_ys:
                if abs(wrist_y - ankle_y) < threshold:
                    wrist_near_ankle = True
                    break  # biri bile yeter

        #print(f"Açı: {avg_angle:.1f} | El-ayak yakın: {wrist_near_ankle} | Eşik: {threshold:.1f}")

        # oturma eşik kontrolü
        if avg_angle < squat_angle_threshold or wrist_near_ankle:
            return "squatting"
        else:
            return "standing"

    except:
        return None

def calculate_velocity(prev_point, current_point):
    """İki nokta arasındaki hızı hesapla"""
    return math.hypot(current_point[0] - prev_point[0], current_point[1] - prev_point[1])

while cap.isOpened():
    ret, frame = cap.read()
    frame_number += 1
    if not ret:
        break

    center_points_cur_frame = [] 
    keypoints_cur_frame = {} 
    boxes_cur_frame = {} # her yeni framede listeler sıfırlanır

    results = model(frame)
    for result in results:
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            center_points_cur_frame.append((cx, cy))
            boxes_cur_frame[(cx, cy)] = (x1, y1, x2, y2)
            class_id = int(box.cls[0])
            label = model.names[class_id]
            if class_id == 0 :
                keypoints = result.keypoints.data.cpu().numpy()
                person_keypoints = keypoints[i]
                keypoints_cur_frame[(cx, cy)] = person_keypoints # merkez noktaya göre anahtar noktaları saklar

    # Tracker algoritması başlangıcı
    if frame_number <= 2: # ilk iki kare için
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1]) # öklid mesafesi bulunur
                if distance < 50:
                    tracking_objects[track_id] = pt
                    tracking_keypoints[track_id] = keypoints_cur_frame.get(pt, []) # Keypoints'i ata
                    tracking_boxes[track_id] = boxes_cur_frame.get(pt, (0, 0, 0, 0)) # Kutuyu ata
                    tracking_pose_status[track_id] = (" ", time.time()) # Durum ata
                    tracking_velocities[track_id] = []  # Hız verilerini başlat
                    track_id += 1
    else:
        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()
        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                if distance < 50:
                    # Hızı hesapla ve kaydet
                    velocity = calculate_velocity(pt2, pt)
                    if object_id not in tracking_velocities:
                        tracking_velocities[object_id] = []
                    tracking_velocities[object_id].append(velocity)
                    # Sadece son N hızı tut
                    if len(tracking_velocities[object_id]) > running_history_size:
                        tracking_velocities[object_id] = tracking_velocities[object_id][-running_history_size:]
                    
                    tracking_objects[object_id] = pt
                    tracking_keypoints[object_id] = keypoints_cur_frame.get(pt, []) # Keypointsi günceller
                    tracking_boxes[object_id] = boxes_cur_frame.get(pt, (0, 0, 0, 0))  # Kutuyu günceller
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                        continue

            if not object_exists: # nesne artık yoksa takip edilenler listesinden çıkartılır
                pose, start_time = tracking_pose_status[object_id]
                duration = time.time() - start_time
                if pose.strip() != "":
                    video_time = frame_number / fps
                    pose_log_records.append([
                        object_id,
                        pose,  # prev_pose yerine pose
                        round(duration, 2),
                        round(video_time,2)
                    ])
                tracking_objects.pop(object_id)
                tracking_keypoints.pop(object_id)
                tracking_boxes.pop(object_id)
                tracking_pose_status.pop(object_id)
                if object_id in tracking_velocities:
                    tracking_velocities.pop(object_id)
                if object_id in running_counters:
                    running_counters.pop(object_id)
                if object_id in person_visible_kpts:
                    person_visible_kpts.pop(object_id)  # iskelet izlerini temizler

        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            tracking_keypoints[track_id] = keypoints_cur_frame.get(pt, []) # Yeni nesne için keypoints ata
            tracking_boxes[track_id] = boxes_cur_frame.get(pt, (0, 0, 0, 0)) # Yeni nesne için kutu ata
            tracking_pose_status[track_id] = (" ", time.time()) # Yeni nesne için durum ata
            tracking_velocities[track_id] = []  # Yeni nesne için hız listesi başlat
            track_id += 1

    for object_id, pt in tracking_objects.items(): # Ekran çıktısı için
        x1, y1, x2, y2 = tracking_boxes[object_id]
        color = person_colors[object_id % len(person_colors)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)
        if object_id in tracking_velocities and tracking_velocities[object_id]:
            avg_velocity = sum(tracking_velocities[object_id]) / len(tracking_velocities[object_id])
            cv2.putText(frame, f"v: {avg_velocity:.1f}", (pt[0], pt[1] + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 5 FRAME'DE BİR POZ TAHMİNİ
        pose_label = " "
        if object_id in tracking_keypoints:
            keypoints = tracking_keypoints[object_id]
            bbox = x1,y1,x2,y2
            if frame_number % 5 == 0: 
                new_pose = analyze_pose(keypoints, bbox, object_id)
                if new_pose is not None:
                    prev_pose, prev_time = tracking_pose_status.get(object_id, ("", time.time()))
                    if new_pose != prev_pose:
                        duration = time.time() - prev_time
                        if prev_pose.strip() != "":
                            video_time = frame_number / fps
                            pose_log_records.append([
                                object_id,
                                prev_pose,
                                round(duration, 2),
                                round(video_time,2)
                            ])
                        tracking_pose_status[object_id] = (new_pose, time.time())
                    pose_label = new_pose
                else:
                    pose_label = tracking_pose_status.get(object_id, (" ", time.time()))[0]
            else:
                pose_label = tracking_pose_status.get(object_id, (" ", time.time()))[0]
            current_time = time.time()
            if object_id not in tracking_pose_status:
                tracking_pose_status[object_id] = (pose_label, current_time)
            else:
                prev_pose, start_time = tracking_pose_status[object_id]
                if pose_label != prev_pose:
                    tracking_pose_status[object_id] = (pose_label, current_time)
                elapsed_time = current_time - tracking_pose_status[object_id][1]
                cv2.putText(frame, f"{pose_label} - {int(elapsed_time)}s", (x1 - 20, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # BBOX İÇERİSİNDE OLMAYAN KEYPOİNTSLERİ SİLMEK İÇİN
            visible_kpts = {}
            for keypoint_index, keypoint in enumerate(keypoints):
                x, y, key_conf = keypoint
                if key_conf > 0.5 and x1 < x < x2 and y1 < y < y2:
                    x,y = int(x), int(y)
                    visible_kpts[keypoint_index] = (x,y)
                    cv2.putText(frame, str(keypoint_index), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.4, color, 1) # Anahtar numaraları işaretle
                    cv2.circle(frame, (int(x), int(y)), 2, color, -1) # Anahtar noktaları işaretle
            person_visible_kpts[object_id] = visible_kpts

    for object_id in tracking_objects.keys():  # iskelet çizmeye yarar
        if object_id in person_visible_kpts:
            visible_kpts = person_visible_kpts[object_id]
            color = person_colors[object_id % len(person_colors)]
            for a, b in SKELETON:
                if a in visible_kpts and b in visible_kpts:
                    pt1 = visible_kpts[a]
                    pt2 = visible_kpts[b]
                    cv2.line(frame, pt1, pt2, color, 2)

    cv2.imshow('Human Pose Detection', frame)
    out.write(frame)
    center_points_prev_frame = center_points_cur_frame.copy()

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()  
cv2.destroyAllWindows()

# CSV'ye kaydetmek için
for object_id, (pose, start_time) in tracking_pose_status.items():
    duration = time.time() - start_time
    if pose.strip() != "":
        video_time = frame_number / fps
        pose_log_records.append([
            object_id,
            pose,
            round(duration, 2),
            round(video_time, 2)
        ])
with open("pose_log_of_video_6.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["person_id", "pose", "duration_sec", "video_time_sec"]) 
    for record in pose_log_records:
        writer.writerow(record)