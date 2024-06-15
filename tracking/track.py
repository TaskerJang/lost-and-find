import argparse
import cv2
import numpy as np
from functools import partial
from pathlib import Path
import torch
from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS
from boxmot.utils.checks import TestRequirements
from tracking.detectors import get_yolo_inferer
from transformers import BlipProcessor, BlipForConditionalGeneration
from sklearn.cluster import KMeans
from collections import Counter
import re

__tr = TestRequirements()
__tr.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.plotting import save_one_box
import os

# BLIP 모델과 프로세서 초기화
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 주요 색상 인식 함수
def get_dominant_color(image, k=4):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((-1, 3))
    clt = KMeans(n_clusters=k)
    clt.fit(image)
    hist = Counter(clt.labels_)
    dominant_color = clt.cluster_centers_[hist.most_common(1)[0][0]]
    return tuple(map(int, dominant_color))

def color_name(rgb_color):
    colors = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "mint": (170, 255, 170)  # mint color
    }
    color_diffs = []
    for color_name, color_rgb in colors.items():
        diff = np.linalg.norm(np.array(rgb_color) - np.array(color_rgb))
        color_diffs.append((diff, color_name))
    return min(color_diffs)[1]

# 사람 중심 좌표 계산 함수
def get_person_center(bboxes):
    person_center_dict = {}
    for bbox in bboxes:
        xc = int((bbox.xyxy[0, 0] + bbox.xyxy[0, 2]) * 0.5)
        yc = int((bbox.xyxy[0, 1] + bbox.xyxy[0, 3]) * 0.5)
        person_center_dict[int(bbox.id.item())] = (xc, yc)
    return person_center_dict

# 가방 중심 좌표 계산 함수
def get_bag_center(bboxes):
    bag_center_dict = {}
    for bbox in bboxes:
        xc = int((bbox.xyxy[0, 0] + bbox.xyxy[0, 2]) * 0.5)
        yc = int((bbox.xyxy[0, 1] + bbox.xyxy[0, 3]) * 0.5)
        bag_center_dict[int(bbox.id.item())] = (xc, yc)
    return bag_center_dict

# 거리 계산 함수
def get_distance(person_center, bag_center):
    return np.sqrt((person_center[0] - bag_center[0])**2 + (person_center[1] - bag_center[1])**2)

# 매칭 함수
def matching(matching_dict_person_bag, person_center_dict, bag_center_dict):
    for person_id, (person_x, person_y) in person_center_dict.items():
        if person_id not in matching_dict_person_bag.keys():
            min_dist = float('inf')
            closest_bag_id = -1

            for bag_id, (bag_x, bag_y) in bag_center_dict.items():
                if bag_id in matching_dict_person_bag.values():
                    continue

                dist = get_distance((person_x, person_y), (bag_x, bag_y))

                if dist < min_dist:
                    min_dist = dist
                    closest_bag_id = bag_id

            if closest_bag_id != -1:
                matching_dict_person_bag[person_id] = closest_bag_id

    return matching_dict_person_bag

# 유실 추적 함수
def lost_tracking(lost_things, matching_dict_person_bag, person_center_dict, bag_center_dict):
    updated_lost_things = lost_things.copy()  # 기존 유실 목록을 복사

    for person_id, bag_id in matching_dict_person_bag.items():
        if person_id not in person_center_dict:
            if bag_id in bag_center_dict:
                if person_id in updated_lost_things:
                    updated_lost_things[person_id]['lost_frame_count'] += 1
                    if updated_lost_things[person_id]['lost_frame_count'] >= 10:  # 유실 조건을 만족하는 프레임 수
                        if not updated_lost_things[person_id]['lost']:
                            print("\n유실물 발생 : bag_id : ", bag_id)
                            updated_lost_things[person_id]['lost'] = True
                            updated_lost_things[person_id]['bag_id'] = bag_id  # 유실된 가방 ID 설정
                else:
                    updated_lost_things[person_id] = {'bag_id': bag_id, 'lost_frame_count': 1, 'lost': False}
        else:
            if person_id in updated_lost_things:
                updated_lost_things[person_id]['lost_frame_count'] = 0
                updated_lost_things[person_id]['lost'] = False

    return updated_lost_things

# 유실 업데이트 함수
def lost_update(person_center_dict, bag_center_dict, lost_things):
    updated_lost_things = lost_things.copy()  # 기존 유실 목록을 복사

    for person_id, person_center in person_center_dict.items():
        if person_id in updated_lost_things:
            updated_lost_things[person_id]['bbox'] = person_center
            updated_lost_things[person_id]['class'] = 'person'
        else:
            updated_lost_things[person_id] = {'bbox': person_center, 'class': 'person', 'lost_frame_count': 0, 'lost': False}
    for bag_id, bag_center in bag_center_dict.items():
        if bag_id not in updated_lost_things:
            updated_lost_things[bag_id] = {'bbox': bag_center, 'class': 'suitcase', 'lost_frame_count': 0, 'lost': False}
        else:
            updated_lost_things[bag_id]['bbox'] = bag_center  # Ensure bbox is updated for lost items

    return updated_lost_things

def capture_lost_item_image(img, bbox):
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    lost_item_image = img[y1:y2, x1:x2]
    return lost_item_image

def capture_lost_item_image_right_half(img, bbox):
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    mid_x = (x1 + x2) // 2
    lost_item_image = img[y1:y2, mid_x:x2]
    return lost_item_image

def generate_description(image):
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    return description

def remove_adjectives_before_item(description):
    pattern = r'(\b(?:red|green|blue|black|white|yellow|cyan|magenta|mint)\b\s*)?(handbag|backpack|suitcase)'
    match = re.search(pattern, description)
    if match:
        return description[match.start(2):]
    return description

def draw_text(img, text, position):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 0, 0)
    thickness = 2
    cv2.putText(img, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

def on_predict_start(predictor, persist=False):
    assert predictor.custom_args.tracking_method in TRACKERS, \
        f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = TRACKER_CONFIGS / (predictor.custom_args.tracking_method + '.yaml')
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.custom_args.tracking_method,
            tracking_config,
            predictor.custom_args.reid_model,
            predictor.device,
            predictor.custom_args.half,
            predictor.custom_args.per_class
        )
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers

@torch.no_grad()
def run(args):
    yolo = YOLO(
        args.yolo_model if 'yolov8' in str(args.yolo_model) else 'yolov8n.pt',
    )

    print(f"Using YOLO model: {args.yolo_model}")
    print(f"Classes to detect: {args.classes}")

    results = yolo.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        show=args.show,
        stream=True,
        device=args.device,
        show_conf=args.show_conf,
        save_txt=args.save_txt,
        show_labels=args.show_labels,
        save=False,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        line_width=args.line_width
    )

    yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))

    if 'yolov8' not in str(args.yolo_model):
        m = get_yolo_inferer(args.yolo_model)
        model = m(
            model=args.yolo_model,
            device=yolo.predictor.device,
            args=yolo.predictor.args
        )
        yolo.predictor.model = model

    yolo.predictor.custom_args = args

    # 프로젝트 경로와 이름을 사용하여 비디오 저장 경로 설정
    project_path = Path(args.project) / args.name
    video_path = project_path / f"{args.name}.mp4"
    video_writer = None

    # 프로젝트 경로가 존재하지 않으면 생성
    if not project_path.exists():
        project_path.mkdir(parents=True, exist_ok=True)

    matching_dict_person_bag = {}
    lost_things = {}

    frame_idx = 0  # 프레임 인덱스 초기화

    for r in results:
        img = yolo.predictor.trackers[0].plot_results(r.orig_img, args.show_trajectories)

        print("BBoxes:", r.boxes)
        print("Classes:", r.boxes.cls)

        person_bboxes = [r.boxes[i] for i in range(len(r.boxes)) if r.boxes.cls[i] == 1]  # 클래스 1: 사람
        bag_bboxes = [r.boxes[i] for i in range(len(r.boxes)) if r.boxes.cls[i] == 0]    # 클래스 0: 가방

        print(f"Person bboxes: {person_bboxes}")
        print(f"Bag bboxes: {bag_bboxes}")

        person_center_dict = get_person_center(person_bboxes)
        bag_center_dict = get_bag_center(bag_bboxes)

        matching_dict_person_bag = matching(matching_dict_person_bag, person_center_dict, bag_center_dict)
        lost_things = lost_tracking(lost_things, matching_dict_person_bag, person_center_dict, bag_center_dict)
        lost_things = lost_update(person_center_dict, bag_center_dict, lost_things)

        print(f"Matching dict: {matching_dict_person_bag}")  # 매칭된 결과를 출력
        print(f"Person centers: {person_center_dict}")       # 사람의 중심 좌표를 출력
        print(f"Bag centers: {bag_center_dict}")             # 가방의 중심 좌표를 출력

        # cv2.line 그리기 전에 img 객체 확인
        print(f"img shape: {img.shape}, dtype: {img.dtype}")

        for person_id, bag_id in matching_dict_person_bag.items():
            if person_id in person_center_dict and bag_id in bag_center_dict:  # person_id와 bag_id가 각 딕셔너리에 있는지 확인
                person_center = person_center_dict[person_id]
                bag_center = bag_center_dict[bag_id]
                print(f"Drawing line: Person ID {person_id} at {person_center} to Bag ID {bag_id} at {bag_center}")
                cv2.line(img, person_center, bag_center, (0, 255, 0), 2)
            else:
                print(f"Skipping line: Person ID {person_id} or Bag ID {bag_id} not found in current frame")

        # 유실물 발생 시 이미지 캡처 및 텍스트 생성
        for person_id, info in lost_things.items():
            if info['lost'] and 'bag_id' in info:
                bag_id = info['bag_id']
                for box in bag_bboxes:
                    if int(box.id.item()) == bag_id:
                        bag_bbox = box.xyxy[0].cpu().numpy()
                        lost_item_image = capture_lost_item_image(img, bag_bbox)
                        right_half_image = capture_lost_item_image_right_half(img, bag_bbox)
                        dominant_color = get_dominant_color(right_half_image)
                        color_name_text = color_name(dominant_color)
                        description = generate_description(lost_item_image)
                        description_cleaned = remove_adjectives_before_item(description)

                        # 특정 키워드가 포함된 경우에만 텍스트를 표시
                        if any(keyword in description_cleaned for keyword in ["handbag", "backpack", "suitcase"]):
                            final_description = f"{color_name_text} {description_cleaned}"
                            draw_text(img, final_description, (int(bag_bbox[0]), int(bag_bbox[1]) - 10))

        # VideoWriter 초기화
        if video_writer is None:
            height, width, _ = img.shape
            video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

        video_writer.write(img)  # 프레임에 그린 선과 바운딩 박스를 포함한 이미지를 VideoWriter에 기록

        frame_idx += 1  # 프레임 인덱스 증가

    if video_writer:
        video_writer.release()  # 모든 프레임을 기록한 후 VideoWriter 객체 해제

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov8n.pt',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    parser.add_argument('--classes', nargs='+', type=int, default=[0, 1],
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track', type=Path,
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--show-trajectories', action='store_true',
                        help='show trajectories')
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')
    parser.add_argument('--agnostic-nms', default=False, action='store_true',
                        help='class-agnostic NMS')

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    run(opt)
