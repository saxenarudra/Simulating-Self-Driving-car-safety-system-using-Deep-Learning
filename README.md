
# Simulating Self-Driving car and safety system using Deep Learning

Simulating the operation of Self-Driving driving car on virtual enviroment with Convolutional Neural Networks(CNN).

To implement this project I will train, validate and test a model using Keras API. 

Input will be camera data from the dataset, and output will be steering angle to autonomous vehicle.

## Architecture of project:-
1) Using simuator to generate dataset with human driving behaviour(errorfree).
2) Design and train a deep Learning model that predicts steering angle with help of raw data.
3) Train and vaidate the model using 80-20 validation set.
4) Complete a lap on simulator track autonomously.

#Dependencies and Libraries used.

I have used anaconda to generate enviroment with exactly neeeded Dependencies.
Make sure you have anaconda installed in order to setup the enviroment

```bash
  # I recommend using TensorFlow with GPU due to heavy processing of project

    conda env create -f environment-gpu.yml
```

Libraries:- 

1) Keras:- This helps dealing with image data using TensorFlow2. Having fast experimentation methods helping to make iterations.

Keras allows aids in using GPU tensor operations which will be necessary to train models on. This scaled models can be trained with multiple layers

2) Matplotlib :- Plotting library to generate graph comparisions, used for comparing training and testing dataset accuray.

3) Numpy :- Important library for mathematical operations and matrices. This library is critical to Keras for performing expressions.

4) Pandas :- Data manipulation and analysis tool for data structure operations.

5) scikit-learn :-  Allows performing classification, regression and clustering operations.


## Simulator and Requirements

Installing Udacity open-source Simulator is necessary component of this project for both data generation and displaying model abilities.

Setup process and dependencies for Simulator:- 
1)	Making sure all setup files are present in local directory, they are hosted on GitHub.
2)	Installing game development engine called Unity which is available for free at host [website](https://unity.com/), it is open source licensed software for use.
3)	Simulator binary file Version 2 is available for [windows](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-windows.zip)/[mac](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-mac.zip)/[linux](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-linux.zip) platforms, it can be ran after installing unity engine and dependencies. 
4)	Large storage drive will be required as three camera input and other data can hold up large space if one full run is performed. 


This project is tested on following Minimum Specification Requirements:-
* Intel i5 dual core processor.
* Minimum 8GB RAM.
* GPU (Nvidia Preferable)
* Atleast 10 GB of free storage.

Run the simulator in training mode and record the runs for 5 iterations and copy recorded data in local directory.

Simulator outputs code in .CSV formats and all image data is stored locally.
![alt text](https://github.com/saxenarudra/Simulating-Self-Driving-car-safety-system-using-Deep-Learning/blob/main/dataset.jpg?raw=true)



## Training model
  Once Simulator is installed just run the binary file and run following command in bash terminal.

``` python model.py ``` 

This will generate a file model-<epoch>.h5
  
  ![alt text](https://github.com/saxenarudra/Simulating-Self-Driving-car-safety-system-using-Deep-Learning/blob/main/TrainAndTest.pngraw=true)

## Running the simulated model


I have also uploaded pretrained model called model.h5 to run using that enter following command:-

``` python drive.py model.h5  ``` 

This will automatically start driving the vehicle which is in automatic mode in simulator.
  
  ![alt text](https://github.com/saxenarudra/Simulating-Self-Driving-car-safety-system-using-Deep-Learning/blob/main/successful%20test.png?raw=true)

## Author
[Rudra Saxena](https://github.com/saxenarudra): Myself Rudra Saxena I am Grad Student at Lakehead University.

## Acknowledgement
Thank you [Dr. Trevor Tomesh](https://github.com/trevortomesh) for help through the project.

