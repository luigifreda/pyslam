# Architecture

Here you can find a couple of diagram sketches that provide an overview of the main system components, and classes relationships and dependencies. **[WIP]** 

---
## SLAM System

<p align="center">
<img src="./images/slam_architecture.png" alt="SLAM"  /> 
</p>


**Note**: You might be wondering why I used **Processes** instead of **Threads** in some cases. The reason is that, at least in Python 3.8 (the version supporting pySLAM), only one thread can execute at a time within a single process due to the Global Interpreter Lock (GIL). On the other hand, using multiprocessing (separate processes that do not share the GIL) enables better parallelism.

---
## Feature Tracker

<p align="center">
<img src="./images/feature_tracker.png" alt="Feature Tracker"  /> 
</p>

---
## Feature Matcher

<p align="center">
<img src="./images/feature_matcher.png" alt="Feature Matcher"  /> 
</p>


---
## Loop Detector 

<p align="center">
<img src="./images/loop_detector.png" alt="Loop Detector"  /> 
</p>

---
## Depth Estimator 

<p align="center">
<img src="./images/depth_estimator.png" alt="Depth Estimator"  /> 
</p>


