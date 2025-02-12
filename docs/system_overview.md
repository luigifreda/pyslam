# System Overview

<!-- TOC -->

- [System Overview](#system-overview)
  - [SLAM Workflow](#slam-workflow)
  - [SLAM Components](#slam-components)
  - [Main System Components](#main-system-components)
    - [Feature Tracker](#feature-tracker)
    - [Feature Matcher](#feature-matcher)
    - [Loop Detector](#loop-detector)
    - [Depth Estimator](#depth-estimator)
    - [Volumetric Integrator](#volumetric-integrator)

<!-- /TOC -->

In this document, you will find some diagram sketches that provide an overview of the main workflow, system components, and class relationships/dependencies. To make the diagrams more readable, some minor components and arrows have been omitted.

---

## SLAM Workflow

<p align="center">
<img src="./images/slam_workflow.png" alt="SLAM Workflow"  /> 
</p>

---
## SLAM Components

<p align="center">
<img src="./images/slam_components.png" alt="SLAM Components"  /> 
</p>


**Note**: In some case, I used **Processes** instead of **Threads** because in Python 3.8 (used by pySLAM) the Global Interpreter Lock (GIL) allows only one thread can execute at a time within a single process. Multiprocessing avoids this limitation and enables better parallelism, though it involves data duplication via pickling. See this nice [post](https://www.theserverside.com/blog/Coffee-Talk-Java-News-Stories-and-Opinions/Is-Pythons-GIL-the-software-worlds-biggest-blunder).


---

## Main System Components

### Feature Tracker

<p align="center">
<img src="./images/feature_tracker.png" alt="Feature Tracker"  /> 
</p>


### Feature Matcher

<p align="center">
<img src="./images/feature_matcher.png" alt="Feature Matcher"  /> 
</p>


### Loop Detector 

<p align="center">
<img src="./images/loop_detector.png" alt="Loop Detector"  /> 
</p>


### Depth Estimator 

<p align="center">
<img src="./images/depth_estimator.png" alt="Depth Estimator"  /> 
</p>


### Volumetric Integrator

<p align="center">
<img src="./images/volumetric_integrator.png" alt="Volumetric Integrator"  /> 
</p>


