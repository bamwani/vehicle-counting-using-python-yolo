# vehicle-counting-using-python-yolo

Vehicle counting in a conjusted traffic road where background subtraction gives lower performance.

![SCREENSHOT](https://github.com/bamwani/vehicle-counting-using-python-yolo/blob/master/Screenshot.png)
ps: The car with-1.5Km/h is actually going in reverse :D

This project use YOLOv3 for Vehicle detection and SORT(Simple Online and Realtime Tracker) for vehicle tracking

# To run the project:

1. Download the code or simply run: ``` git clone https://github.com/bamwani/vehicle-counting-using-python-yolo ``` in the terminal

2. Make sure you change the line of detection according to your video and fine tune the threshold and confidence for YOLO model

2. Run ```main.py -input /path/to/video/file.avi -output /path/for/output/file.avi -yolo /path/to/YOLO/directory/``` 



I will keep making commits to improve the speed detection of vehicles. I have tried Kalman filter but it fails to work in conjusted traffic and side angle positioned cameras.



# References:


[YOLO](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)

[SORT Algorithm](https://github.com/abewley/sort)

[Reference](https://github.com/guillelopez/python-traffic-counter-with-yolo-and-sort)
