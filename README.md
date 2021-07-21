### Objective

#### When distance between two persons is lesser than specified amount they are said to be violating. Here We are identifying the pairs of person found violating the min distance condition in a surveillance video 
#### When distance between two persons is lesser than specified amount(on pixel plane) they are said to be violating. Here We are identifying the pairs of person found violating the min distance condition in a surveillance video 
### Libraries used
* [pafy](https://pypi.org/project/pafy/) -: for getting youtube video frame by frame
* [opencv](https://pypi.org/project/opencv-python/) -: for drawing rectangles and lines. And displaying each frame
* [imutils](https://pypi.org/project/imutils/) -: for resizing the frame
* [numpy](https://numpy.org/) -: for performing basic algebraic/multi-dimensional array operations
### Object Detection algorithm used [YOLOv3](https://pjreddie.com/darknet/yolo/)
### Quick overview
![YOLO v3 project image](https://user-images.githubusercontent.com/62443378/125178033-8688ff00-e1fe-11eb-8965-ae7478698a7a.jpeg)
### YOLOv3 brief overview 
#### ![YOLOv3_ Real-Time Object Detection Algorithm (What's New_) _ viso ai - Brave 7_4_2021 7_57_48 PM](https://user-images.githubusercontent.com/62443378/124389295-19f49880-dd04-11eb-928e-3f555d7633a5.png)
#### YOLOv3 (You Only Look Once, Version 3) is a real-time object detection algorithm that identifies specific objects in videos, live feeds, or images. Versions 1-3 of YOLO were created by Joseph Redmon and Ali Farhadi.
#### After a frame is read from the input image or video stream, it is passed through the blobFromImage function to convert it to an input [blob](https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/) for the neural network. In this process, it scales the image pixel values to a target range of 0 to 1 using a scale factor of 1/255. It also resizes the image to the given size of (416, 416) without cropping. Note that we do not perform any mean subtraction here, hence pass [0,0,0] to the mean parameter of the function and keep the swapRB parameter to its default value of 1.
```python
blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
net.setInput(blob)
```
#### The output blob is then passed in to the network as its input and a forward pass is run to get a list of predicted bounding boxes as the network’s output. These boxes go through a post-processing step in order to filter out the ones with low confidence scores and boex containing objects other than Person.
#### The forward function in OpenCV’s Net class needs the ending layer till which it should run in the network. Since we want to run through the whole network, we need to identify the last layer of the network. We do that by using the function getUnconnectedOutLayers() that gives the names of the unconnected output layers, which are essentially the last layers of the network which we pass to the function as ln
```python
ln = net.getLayerNames()                                    
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]  
```
``` python
layerOutputs = net.forward(ln)   
```
### Sample frames from output
![detection output-3](https://user-images.githubusercontent.com/62443378/125177918-745a9100-e1fd-11eb-8d22-28e7b1d472b4.png)
![detection-output-2](https://user-images.githubusercontent.com/62443378/125177894-38273080-e1fd-11eb-9b69-4ce3048e865f.png)
### Other uses of similar technology
* Video surveillance - Because state-of-the-art object detection techniques can accurately identify and track multiple instances of a given object in a scene, these techniques naturally lend themselves to automating video surveillance systems. For instance, object detection models are capable of tracking multiple people at once, in real-time, as they move through a given scene or across video frames. From retail stores to industrial factory floors, this kind of granular tracking could provide invaluable insights into security, worker performance and safety, retail foot traffic, and more.
* Crowd counting - Crowd counting is another valuable application of object detection. For densely populated areas like theme parks, malls, and city squares, object detection can help businesses and municipalities more effectively measure different kinds of traffic—whether on foot, in vehicles, or otherwise. This ability to localize and track people as they maneuver through various spaces could help businesses optimize anything from logistics pipelines and inventory management, to store hours, to shift scheduling, and more. Similarly, object detection could help cities plan events, dedicate municipal resources, etc.
### References -: 
#### https://learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/
#### https://viso.ai/deep-learning/yolov3-overview/
#### Example surveillance video used - https://youtu.be/2bKXv_XviFc
