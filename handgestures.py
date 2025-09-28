from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2 as cv
import numpy as np
from datetime import datetime
import multiprocessing as mp
import time
from pose import Pose, Part, PoseSequence
from memryx import AsyncAccl
import sys
import os
from pathlib import Path
from queue import Queue, Empty
import torchvision.ops as ops
from typing import List
from multiprocessing import Value
from tracker.byte_tracker import BYTETracker
from argparse import Namespace
import torch

# Initialize FastAPI and Jinja2Templates
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="./templates")

# Add favicon route to prevent 404 errors
@app.get("/favicon.ico")
async def favicon():
    return JSONResponse({"message": "No favicon"}, status_code=404)

# Global variables
globexercise = 'bicepcurl'
run_start_time = None
vid_rec_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
frame_queue = mp.Queue()  # Multiprocessing Queue for frame sharing
inference_process = None
poseapp = None
run_flag = Value('b', False) 

class poseApp:
    COLOR_LIST = list([[128, 255, 0], [255, 128, 50], [128, 0, 255], [255, 255, 0],
                   [255, 102, 255], [255, 51, 255], [51, 153, 255], [255, 153, 153],
                   [255, 51, 51], [153, 255, 153], [51, 255, 51], [0, 255, 0],
                   [255, 0, 51], [153, 0, 153], [51, 0, 51], [0, 0, 0],
                   [0, 102, 255], [0, 51, 255], [0, 153, 255], [0, 153, 153]])
        
    # Define keypoint pairs for drawing skeletons
    KEYPOINT_PAIRS = [
            (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), (6, 8),
            (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]
    
    EXERCISE_CFG = {
                    "bicep_curl" : {"up_angle": 5, "down_angle":145, "l_parts": ['lshoulder','lelbow', 'lwrist'], "r_parts": ['rshoulder','relbow', 'rwrist'], "text": "Make sure atleast one arm and your hip is visible"}
                 }
    BYTETRACK_CFG = {
                    "track_thresh" : 0.5, # High_threshold
                    "track_buffer" : 50, # Number of frame lost tracklets are kept
                    "match_thresh": 0.9,    # Matching threshold for bounding boxes
                    "track_buffer": 30,     # Number of frames to keep track without new detection
                    "mot20": True          # Use settings for MOT20 dataset (if using MOT20)
                    }
    
    def __init__(self, dfp, post_model, model_input_shape, frame_queue, run_flag, exercise, mirror=False):
        # self.cam = cam
        self.cam = cv.VideoCapture(0)
        if not self.cam.isOpened():
            print("Unable to read camera feed")
            return
        self.cam.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
        self.cam.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
        self.input_height = int(self.cam.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.input_width = int(self.cam.get(cv.CAP_PROP_FRAME_WIDTH))
        print("CAM Height and Width ", self.input_height, self.input_width)
        self.model_input_shape = model_input_shape
        self.capture_queue = Queue(maxsize=5)
        # self.pose_output_queue = deque(maxlen=2)
        self.mirror = mirror
        self.box_score = 0.25
        self.kpt_score = 0.5
        self.nms_thr = 0.2
        self.running = run_flag
        self.ratio = None
        self.output_frame_queue = frame_queue
        self.accl = None
        self.dfp = dfp
        self.post_model = post_model
        self.exercise = exercise
        
        # Tracking arrays for multiple people
        self.angles = []
        self.last_stage_change = []
        
        # Time tracking variables - per person
        self.total_condition_time = []  # Total time condition is met per person
        self.condition_start_time = []  # When condition started being met (None if not currently met)
        self.is_condition_met = []      # Current state of condition per person
        self.condition_ratios = []      # Ratio of condition time to total time per person
        
        # Global timing
        self.camera_start_time = time.time()  # When camera started running
        
        tracker_config = Namespace(**self.BYTETRACK_CFG)
        self.tracker = BYTETracker(tracker_config, frame_rate=30)
        self.tracked_objects = None
        self.searchtext = ""
        self.searchtxtpos = (int(self.input_width/2), int(self.input_height/2))
        
        def cal_image_center():
            text = "Searching for people"
            font = cv.FONT_HERSHEY_SIMPLEX
            scale = 2
            thickness = 2
            # Get the text size
            (text_width, text_height), baseline = cv.getTextSize(text, font, scale, thickness)
            # Calculate the center position
            x = (self.input_width - text_width) // 2
            y = (self.input_height + text_height) // 2
            self.searchtxtpos = (x, y)
            
        cal_image_center()
    
    def generate_frame(self):
        while True:
            ok, frame = self.cam.read()
            # print(frame.shape)
            if not self.running.value:
                print('EOF')
                # self.cam.release() 
                self.stop()
                return None
            else:
                if self.capture_queue.full():
                    # drop frame
                    continue
                else:
                    if self.mirror:
                        frame = cv.flip(frame, 1)
                    self.capture_queue.put(frame)
                    out, self.ratio = self.preprocess_image(frame)
                    return out
    
    def preprocess_image(self, image):
        h, w = image.shape[:2]
        r = min(self.model_input_shape[0] / h, self.model_input_shape[1] / w)
        image_resized = cv.resize(image, (int(w * r), int(h * r)), interpolation=cv.INTER_LINEAR)
        # Create padded image
        padded_img = np.full((self.model_input_shape[0], self.model_input_shape[1], 3), 114, dtype=np.uint8)
        padded_img[:int(h * r), :int(w * r)] = image_resized
        # Normalize to [0, 1]
        padded_img = (padded_img / 255.0).astype(np.float32)
        # Add batch dimension, and move to channel first format as expected by onnx model
        padded_img = np.expand_dims(padded_img, axis=0)
        padded_img = np.transpose(padded_img, (0, 3, 1, 2))
        return padded_img, r
    
    def xywh2xyxy(self, box: np.ndarray) -> np.ndarray:
        box_xyxy = np.copy(box)
        box_xyxy[:, 0] = box[:, 0] - box[:, 2] / 2
        box_xyxy[:, 1] = box[:, 1] - box[:, 3] / 2
        box_xyxy[:, 2] = box[:, 0] + box[:, 2] / 2
        box_xyxy[:, 3] = box[:, 1] + box[:, 3] / 2
        return box_xyxy
    
    def compute_iou(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])
        inter_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - inter_area
        return inter_area / union_area
    
    def nms_process(self, boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
        # Apply non-maximum suppression to reduce redundant overlapping boxes
        sorted_idx = np.argsort(scores)[::-1]  # Sort scores in descending order
        keep_idx = []
        while sorted_idx.size > 0:
            idx = sorted_idx[0]  # Keep the box with the highest score
            keep_idx.append(idx)
            ious = self.compute_iou(boxes[idx, :], boxes[sorted_idx[1:], :])  # Calculate IoU
            rest_idx = np.where(ious < iou_thr)[0]  # Keep boxes with IoU below threshold
            sorted_idx = sorted_idx[rest_idx + 1]
        return keep_idx
    
    def update_condition_timer(self, index):
        """Update the timer for how long the angle condition is met for a specific person"""
        current_time = time.time()
        
        # Check if condition is currently met
        condition_met = (self.EXERCISE_CFG[self.exercise]["up_angle"] < self.angles[index] < self.EXERCISE_CFG[self.exercise]["down_angle"])
        
        # If condition just started being met
        if condition_met and not self.is_condition_met[index]:
            self.condition_start_time[index] = current_time
            self.is_condition_met[index] = True
            
        # If condition was met and is still met, add elapsed time
        elif condition_met and self.is_condition_met[index]:
            # Continuously accumulate time while condition is met
            elapsed_since_start = current_time - self.condition_start_time[index]
            # Reset the start time to current time to avoid double counting
            self.total_condition_time[index] += elapsed_since_start
            self.condition_start_time[index] = current_time
            
        # If condition just stopped being met
        elif not condition_met and self.is_condition_met[index]:
            # Add the final elapsed time
            elapsed_since_start = current_time - self.condition_start_time[index]
            self.total_condition_time[index] += elapsed_since_start
            self.is_condition_met[index] = False
            self.condition_start_time[index] = None
    
    def get_total_camera_time(self):
        """Get total time the camera has been running"""
        return time.time() - self.camera_start_time
    
    def get_current_condition_time(self, index):
        """Get current total condition time including any ongoing period"""
        total_time = self.total_condition_time[index]
        
        # If condition is currently being met, add the ongoing time
        if self.is_condition_met[index] and self.condition_start_time[index] is not None:
            total_time += time.time() - self.condition_start_time[index]
            
        return total_time
    
    def get_condition_ratio(self, index):
        """Calculate ratio of condition time to total camera time"""
        condition_time = self.get_current_condition_time(index)
        total_camera_time = self.get_total_camera_time()
        
        # Avoid division by zero
        if total_camera_time == 0:
            return 0.0
        
        ratio = condition_time / total_camera_time
        return ratio
    
    def process_model_output(self, *ofmaps):
        try:
            img = self.capture_queue.get()
        except Empty:
            return None
        self.capture_queue.task_done()
        
        # Process model output (keypoints and bounding boxes)
        predict = ofmaps[0].squeeze(0).T  # Shape: [8400, 56]
        predict = predict[predict[:, 4] > self.box_score, :]  # Filter boxes by confidence score
        scores = predict[:, 4]
        boxes = predict[:, 0:4] / self.ratio
        boxes = self.xywh2xyxy(boxes)  # Convert bounding box format
        
        # Process keypoints
        kpts = predict[:, 5:]
        for i in range(kpts.shape[0]):
            for j in range(kpts.shape[1] // 3):
                if kpts[i, 3*j+2] < self.kpt_score:  # Filter keypoints by confidence score
                    kpts[i, 3*j: 3*(j+1)] = [-1, -1, -1]
                else:
                    kpts[i, 3*j] /= self.ratio
                    kpts[i, 3*j+1] /= self.ratio 
        
        idxes = self.nms_process(boxes, scores, self.nms_thr)  # Apply NMS
        result = {'boxes': boxes[idxes, :].astype(int).tolist(),
                'kpts': kpts[idxes, :].astype(float).tolist(),
                'scores': scores[idxes].tolist()}
        
        # Draw keypoints and bounding boxes on the image
        boxes, kpts, scores = result['boxes'], result['kpts'], result['scores']
        
        byte_detections = []
        for kpt, score, bbox in zip(kpts, scores, boxes):
            x_min, y_min, x_max, y_max = bbox
            byte_detections.append([x_min, y_min, x_max, y_max, score, 0.7, 0.7, 0])
        
        # Apply tracker
        try:
            self.tracked_objects = self.tracker.update(torch.tensor(byte_detections))
            # Adding new human
            if len(self.tracked_objects) > 0:
                if len(self.tracked_objects) > len(self.angles):
                    new_human = len(self.tracked_objects) - len(self.angles)
                    self.angles += [0] * new_human
                    self.last_stage_change += [time.time()] * new_human
                    # Initialize timing variables for new people
                    self.total_condition_time += [0.0] * new_human
                    self.condition_start_time += [None] * new_human
                    self.is_condition_met += [False] * new_human
                    self.condition_ratios += [0.0] * new_human
                
                # enumerate over track
                for index, track in enumerate(self.tracked_objects):
                    track_id = track.track_id
                    kpt = kpts[index]
                    box = boxes[index]
                    text_position = (box[2] - box[0], box[3] - box[1])
                    
                    for pair in self.KEYPOINT_PAIRS:
                        pt1 = kpt[3 * pair[0]: 3 * (pair[0] + 1)]
                        pt2 = kpt[3 * pair[1]: 3 * (pair[1] + 1)]
                        if pt1[2] > 0 and pt2[2] > 0:
                            cv.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255, 255, 255), 3)
                    
                    # Draw individual keypoints
                    parts_kpt = []
                    for idx in range(len(kpt) // 3):
                        x, y, score = kpt[3*idx: 3*(idx+1)]
                        parts_kpt.append([x, y, score])
                        if score > 0:
                            cv.circle(img, (int(x), int(y)), 5, self.COLOR_LIST[idx % len(self.COLOR_LIST)], -1)
                    
                    pose_cl = Pose(parts_kpt)
                    right, left = self.is_necessary_kpt_visible(pose_cl)
                    
                    if not right and not left:
                        cv.putText(img=img, text=f'ID: {track_id}', org=(text_position[0], text_position[1] - 20), 
                            fontFace=cv.FONT_HERSHEY_DUPLEX,
                            fontScale=0.6,
                            color=(0, 255, 0),
                            thickness=2
                            )
                        cv.putText(img=img, text="Keypoints not visible", org=text_position, 
                                fontFace=cv.FONT_HERSHEY_DUPLEX,
                                fontScale=0.6,
                                color=(0, 0, 255),
                                thickness=2
                                )
                    else:
                        if right:
                            a = pose_cl.get_part_xy(self.EXERCISE_CFG[self.exercise]['r_parts'][0])
                            b = pose_cl.get_part_xy(self.EXERCISE_CFG[self.exercise]['r_parts'][1])
                            c = pose_cl.get_part_xy(self.EXERCISE_CFG[self.exercise]['r_parts'][2])
                            self.angles[index] = self.estimate_pose_angle(a, b, c)
                            text_position = (int(b[0]), int(b[1]))
                        else:
                            a = pose_cl.get_part_xy(self.EXERCISE_CFG[self.exercise]['l_parts'][0])
                            b = pose_cl.get_part_xy(self.EXERCISE_CFG[self.exercise]['l_parts'][1])
                            c = pose_cl.get_part_xy(self.EXERCISE_CFG[self.exercise]['l_parts'][2])
                            self.angles[index] = self.estimate_pose_angle(a, b, c) 
                            text_position = (int(b[0]), int(b[1]))       
                        
                        # Update condition timer
                        self.update_condition_timer(index)
                        
                        # Update the stored ratio
                        self.condition_ratios[index] = self.get_condition_ratio(index)
                        
                        # Get current condition time for display
                        current_condition_time = self.get_current_condition_time(index)
                        total_camera_time = self.get_total_camera_time()
                        condition_ratio = self.condition_ratios[index]
                        
                        # Create display text
                        condition_txt = f'Condition Time: {current_condition_time:.1f}s'
                        camera_txt = f'Total Runtime: {total_camera_time:.1f}s'
                        ratio_txt = f'Efficiency: {condition_ratio*100:.1f}%'
                        
                        cv.putText(img=img, text=f'ID: {track_id}', org=(text_position[0], text_position[1] - 80), 
                            fontFace=cv.FONT_HERSHEY_DUPLEX,
                            fontScale=0.6,
                            color=(0, 255, 0),
                            thickness=2
                            )
                        cv.putText(img=img, text=condition_txt, org=(text_position[0], text_position[1] - 60), 
                                    fontFace=cv.FONT_HERSHEY_DUPLEX,
                                    fontScale=0.6,
                                    color=(0, 255, 0),
                                    thickness=2
                                    )
                        cv.putText(img=img, text=camera_txt, org=(text_position[0], text_position[1] - 40), 
                                    fontFace=cv.FONT_HERSHEY_DUPLEX,
                                    fontScale=0.6,
                                    color=(255, 255, 0),
                                    thickness=2
                                    )
                        cv.putText(img=img, text=ratio_txt, org=(text_position[0], text_position[1] - 20), 
                                    fontFace=cv.FONT_HERSHEY_DUPLEX,
                                    fontScale=0.6,
                                    color=(255, 0, 255),
                                    thickness=2
                                    )
            else:
                self.searchtext = "" if self.searchtext == "Searching for People" else "Searching for People"
                cv.putText(img=img, text=self.searchtext, org=self.searchtxtpos, 
                                    fontFace=cv.FONT_HERSHEY_DUPLEX,
                                    fontScale=2,
                                    color=(0, 0, 255),
                                    thickness=2
                                    )
        except Exception as e:
            print(f"An error occurred while updating tracker: {e}")
            self.tracked_objects = None  # Or set to a default value if necessary
            self.running.value = False
        
        # self.show(img)
        ret, buffer = cv.imencode('.jpg', img)
        self.output_frame_queue.put(buffer)
        
        return img
    
    def estimate_pose_angle(self, a, b, c):
        """
        Calculate the pose angle for object.
        Args:
            a (float) : The value of pose point a
            b (float): The value of pose point b
            c (float): The value o pose point c
        Returns:
            angle (degree): Degree value of angle between three points
        """
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle
    
    def is_necessary_kpt_visible(self, pose):
        right_present = pose.if_given_parts_exists(self.EXERCISE_CFG[self.exercise]['r_parts'])
        left_present = pose.if_given_parts_exists(self.EXERCISE_CFG[self.exercise]['l_parts'])
        return right_present, left_present
        
    def stop(self):
        print("Stopping the poseApp")
        # write results from child process before releasing resources
        try:
            self.write_results_to_file("curl_results.json")
        except Exception as e:
            print(f"Error writing results in child stop: {e}")
        self.cam.release()  # Release camera resources
        while not self.output_frame_queue.empty():
            print("flushing queue in child process ")
            self.output_frame_queue.get()
    
    def start(self):
        if not self.running.value or not self.cam.isOpened():
            self.cam = cv.VideoCapture(0)
            self.running.value = True
            self.camera_start_time = time.time()  # Reset camera start time
            print("Reopened camera")
        # run_start_time is set in start_inference (main process)
        self.accl = AsyncAccl(self.dfp)
        self.accl.set_postprocessing_model(self.post_model, model_idx=0)
        self.accl.connect_input(self.generate_frame)
        self.accl.connect_output(self.process_model_output)
        self.accl.wait()
    
    def write_results_to_file(self, filename="curl_results.json"):
        """Write total runtime and per-person condition time to a JSON file."""
        import json
        global run_start_time
        results = {}
        end_time = time.time()
        total_run = None
        if run_start_time is not None:
            total_run = end_time - run_start_time
        
        # Finalize any ongoing condition periods
        for i in range(len(self.total_condition_time)):
            if self.is_condition_met[i] and self.condition_start_time[i] is not None:
                elapsed_time = end_time - self.condition_start_time[i]
                self.total_condition_time[i] += elapsed_time
                self.condition_start_time[i] = None
                self.is_condition_met[i] = False
        
        results['total_run_time_s'] = total_run
        results['total_camera_time_s'] = end_time - self.camera_start_time
        results['per_person_condition_time_s'] = self.total_condition_time
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f)
            print(f"Wrote results to {filename}")
        except Exception as e:
            print(f"Failed to write results: {e}")

def run_mxa(_run_flag, p_poseapp):
    global run_flag
    # Continuously process frames and add them to the queue
    p_poseapp.start()
    print("Pose Application started")
    # Sleep for safety when process ends
    # while run_flag.value:
    #     time.sleep(1)

def display_pose():
    global frame_queue, run_flag
    while run_flag.value:
        try:
            buffer = frame_queue.get(timeout=1)  # Add timeout to avoid blocking indefinitely
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Empty:
            # Skip frame if the queue is empty to avoid blocking
            continue

def start_inference(exercise):
    global inference_process, frame_queue, poseapp, run_flag, run_start_time
    run_flag.value = True
    run_start_time = time.time()  # Set the start time
    poseapp = poseApp("../models/YOLO_v8_medium_pose_640_640_3_onnx.dfp", "../models/YOLO_v8_medium_pose_640_640_3_onnx_post.onnx", (640, 640), frame_queue, run_flag, exercise, mirror=False)
    if inference_process is None or not inference_process.is_alive():
        inference_process = mp.Process(
            target=run_mxa,
            args=(run_flag, poseapp),
            daemon=True
        )
        inference_process.start()

def stop_inference():
    global inference_process, poseapp, run_flag, frame_queue
    run_flag.value = False
    if not frame_queue.empty():
        while not frame_queue.empty():
            print("flushing in main process ")
            frame_queue.get()
    print("Stop inference is called \n")
    if poseapp is not None:
        print("pose app stop is called\n")
    # Write results to file on stop
        poseapp.write_results_to_file("curl_results.json")
        time.sleep(1)  # Short delay to ensure resources are released
    if inference_process is not None:
        print("Joining the process ")
        inference_process.join()  # Wait a short time for it to join
        inference_process = None

@app.get("/", response_class=HTMLResponse)
@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    # Start bicep curl inference immediately when home page is accessed
    if not run_flag.value:
        start_inference("bicep_curl")
    return templates.TemplateResponse("bicepcurl.html", {"request": request})

@app.get("/bicepcurl", response_class=HTMLResponse)
async def bicepcurl(request: Request):
    if run_flag.value:
        stop_inference()
    else:
        start_inference("bicep_curl")
    return templates.TemplateResponse("bicepcurl.html", {"request": request})

@app.get('/video_rec/{exercise}')
async def video_rec(exercise: str):
    return StreamingResponse(display_pose(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get('/stop_inference')
async def stop_inference_route():
    stop_inference()
    return JSONResponse({"status": "stopped"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
