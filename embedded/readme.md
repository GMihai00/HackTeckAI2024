# Introduction

To enable the object we made a non-conventional approach suited for embedded devices, focusing on image processing speed. The code runs on 6 threads (including the main one) from which one can be optionally disabled. They have the following scopes:

- reading images from camera
- prepossessing images and detecting movement
- detecting bins within the moving objects
- rendering the video content

## Movement detection
As mentioned above the first step we do is 
detect moving objects within the video. To do this we take 2 consecutive frames and apply the following operations:

- convert them to gray-scale
- apply gaussian blur
- get their absolute difference 
- calculate the object thresholds
- repeatedly apply dialate and errode to fill in bounding box of the image tracked.

Due to camera being mobile we also applied an additional filter to track only big enough moving objects and filter out the amount of data.

```python
def is_minimum_obj_size(obj):
    return (obj and obj.get_area() > 4500 and 0.2 < obj.get_aspect_ratio() < 4.0 and
        obj.get_width() > 130 and obj.get_height() > 130 and
        obj.get_diagonal_size() > 200.0 and
        obj.get_contour_area_ratio() > 0.75)
```

The second step we did was to track the objects, we took it's bounding box, calculated its center point and attempted to approximate its next position within the frames.

```python
def predict_next_position(self):
        n = len(self.center_positions)
        if n == 0:
            return
        elif n == 1:
            self.future_position = self.center_positions[-1]
        else:
            delta_x = (self.center_positions[-1][0] * (n - 1) - self.sum_center_pos[0]) / ((n * (n - 1)) / 2.0)
            delta_y = (self.center_positions[-1][1] * (n - 1) - self.sum_center_pos[1]) / ((n * (n - 1)) / 2.0)
            self.future_position = (int(self.center_positions[-1][0] + delta_x),
                                    int(self.center_positions[-1][1] + delta_y))
                                    
```

To prevent losing track of the objects or even counting 2 separate moving objects as one we added some hyperparameter that we tuned to fit our videos. They represent the number of frames we can lose track of the object and the max number of consecutive frames we can have the same moving object in.

```python
MAX_OBJECTS_STORED = 300
MAX_FRAMES_WITHOUT_A_MATCH = 90
```

## Detecting bins

The next step to take is to detect the bins within the frame. To do so, we take the bounding boxes of the moving objects. Like this we cut down on the number of pixels processed, contributing a huge deal to the processing speed of the video. We used the pre-training YOLOV8 model provided within the challenge. We consider bins anything above the threshold of 70% confidence. We tried going lower, but the model was hallucinating in some scenarios. Also, as an additional check, to confidently classify moving objects as bins, we added yet another threshold on the number of consecutive detections required (on different frames).

```python
MINIMUM_CONSECUTIVE_OBJECT_MATCHES = 4
```

## Additional performance improvements

To avoid the cases when the camera is obstructed by other objects, like the case when the trash bins are in the middle of being emptied, we added a check that calculates the similarity of pixels within the frame. If it is above a give threshold, we consider the camera was obstructed and skip processing the frame, cutting it from the processing all together. 


## Saving data

Once a trash bin is detected we save the start and end timestamp of the bin processing inside a csv file. For this we had 2 approaches:

- Calculating the time within the video, as it is saved on disk
- Additional logic that extract exact timestamp within the video, using gcd, for the cameras that have this additional feature. By default, this option is disabled due to adding additional computations, lowering the performance


# Running the code

For a sample run, go to the head of the repo and run: 

```ps1
python ./embedded/main.py
--video_path "<path>"

```

## Enabling additional feature flags

For enabling additional features, please consult the man, using "-h" or "--help" flag

```ps1
python ./embedded/main.py -h
usage: main.py [-h] --video_path VIDEO_PATH [--enable_ocr ENABLE_OCR]
               [--draw_moving DRAW_MOVING] [--render_video RENDER_VIDEO]
               [--render_post_processed_video RENDER_POST_PROCESSED_VIDEO]

options:
  -h, --help            show this help message and exit
  --video_path VIDEO_PATH
                        Path to video file to be processed
  --enable_ocr ENABLE_OCR
                        Enable additional ocr processing of timestamp within video
  --draw_moving DRAW_MOVING
                        Enable flag to draw all moving objects bounding boxes
  --render_video RENDER_VIDEO
                        If enabled display video content, to be disabled when
                        running in prod
  --render_post_processed_video RENDER_POST_PROCESSED_VIDEO
                        Learning flag, to understand the applied video processing
                        operations.
```




