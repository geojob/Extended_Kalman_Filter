#!/usr/bin/env python3

# Columbia Engineering
# MECS 4602 - Fall 2018	

import math
import numpy
import time

import rospy



from state_estimator.msg import RobotPose
from state_estimator.msg import SensorData

class Estimator(object):
    def __init__(self):

        # Publisher to publish state estimate
        self.pub_est = rospy.Publisher("/robot_pose_estimate", RobotPose, queue_size=1)

        # Initial estimates for the state and the covariance matrix
        self.x = numpy.zeros((3,1))
        self.P = numpy.zeros((3,3))

        # Covariance matrix for process (model) noise
        self.V = numpy.zeros((3,3))
        self.V[0,0] = 0.0025
        self.V[1,1] = 0.0025
        self.V[2,2] = 0.005

        self.step_size = 0.01

        # Subscribe to command input and sensory output of robot
        rospy.Subscriber("/sensor_data", SensorData, self.sensor_callback)
        
    # This function gets called every time the robot publishes its control 
    # input and sensory output. You must make use of what you know about 
    # extended Kalman filters to come up with an estimate of the current
    # state of the robot and covariance matrix.
    # The SensorData message contains fields 'vel_trans' and 'vel_ang' for
    # the commanded translational and rotational velocity respectively. 
    # Furthermore, it contains a list 'sens.readings' of the landmarks the
    # robot can currently observe
    def estimate(self, sens):

        #### ----- YOUR CODE GOES HERE ----- ####
        
        #initial prediction based on dynamic model
        pred_state  = [self.x[0] + self.step_size*sens.vel_trans*math.cos(self.x[2]),self.x[1] + self.step_size*sens.vel_trans*math.sin(self.x[2]),self.x[2] + self.step_size*sens.vel_ang]
        
        #F Matriax jacobian (derivative wrt x, y,theta)
        predmod_jac = [[1,0,-self.step_size*sens.vel_trans*math.sin(self.x[2])], [0,1,self.step_size*sens.vel_trans*math.cos(self.x[2])],[0,0,1]]
        
        #initialize landmark x and y position vectors
        lm_x = []
        lm_y = []
        
        # Fill in landmark x and y position vectors
        for i in range(len(sens.readings)):
             
             lm_x.append(sens.readings[i].landmark.x)
             lm_y.append(sens.readings[i].landmark.y)
        
       # print(lm_x)
       # print(lm_y)
       # print(len(sens.readings)) 
       
             
        # remove landmarks that are too close (avoid singularities) First implementation
        # did not work as the index subscript would be out of range due to readings vector directly being modified. Different subscripts for landmark vectors helped solve issue.
        #for i in range(len(sens.readings)):
        
         #    if(numpy.sqrt((pred_state[0][0] - lm_x[i])**2 + (pred_state[1][0] - lm_y[i])**2) < 0.1):
                
          #      del lm_x[i]
          #      del lm_y[i]
          #      del sens.readings[i] 
        # Removing landmarks that are too close (2nd implementation)  
        z=0
        for i in range(len(sens.readings)):
        	
                if numpy.sqrt((pred_state[0][0] - lm_x[z])**2+(pred_state[1][0] - lm_y[z])**2) < 0.1:
                    del sens.readings[i]
                    del lm_x[i]
                    del lm_y[i]
                    z-=1
                z+=1
       
                
        covar_pred = numpy.dot(numpy.dot(predmod_jac, self.P), numpy.transpose(predmod_jac)) + self.V
        
        sensmod_jac = numpy.empty((2*len(sens.readings),3))
        sensmod_var = numpy.zeros((2*len(sens.readings),2*len(sens.readings)))
        y = numpy.empty((2*len(sens.readings),1))
        yhat = numpy.empty((2*len(sens.readings),1))
        innov = numpy.empty((2*len(sens.readings),1))
        # Fill in the sensor model, sensor model covariance, sensor readings and predicted sesor readings
        for i in range(len(sens.readings)):
        
             y[2*i][0] = sens.readings[i].range
             y[2*i+1][0] = sens.readings[i].bearing
             
             yhat[2*i][0] = math.sqrt((lm_x[i] - pred_state[0][0])**2 + (lm_y[i] - pred_state[1][0])**2)
             yhat[2*i+1][0] = math.atan2((lm_y[i] - pred_state[1][0]), (lm_x[i] - pred_state[0][0])) - pred_state[2][0]
             
             #derivative of range wrt x, y, theta
             sensmod_jac[2*i][0] = (pred_state[0][0] - lm_x[i])/(math.sqrt((lm_x[i] - pred_state[0][0])**2 + (lm_y[i] - pred_state[1][0])**2))
             sensmod_jac[2*i][1] = (pred_state[1][0]  - lm_y[i])/(math.sqrt((lm_x[i] - pred_state[0][0])**2 + (lm_y[i] - pred_state[1][0])**2))
             sensmod_jac[2*i][2] = 0
             
             #derivative of bearing wrt x, y, theta
             sensmod_jac[2*i+1][0] = (lm_y[i] - pred_state[1][0])/((lm_x[i] - pred_state[0][0])**2 + (lm_y[i] - pred_state[1][0])**2)
             sensmod_jac[2*i+1][1] = (-lm_x[i] + pred_state[0][0])/((lm_x[i] - pred_state[0][0])**2 + (lm_y[i] - pred_state[1][0])**2)
             sensmod_jac[2*i+1][2] = -1
             
             #covariance of range 0.1 and bearing 0.05
             sensmod_var[2*i][2*i] = 0.1
             sensmod_var[2*i+1][2*i+1] = 0.05
             
             #Compute the innovation between sensor readings and predicted sensor readings
             innov[2*i][0] = y[2*i][0] - yhat[2*i][0]
             innov[2*i+1][0] = y[2*i+1][0] - yhat[2*i+1][0]
             #print(innov)
             
             #ensure bearing differences <pi and >-pi
             while innov[2*i+1][0] > numpy.pi:
                  innov[2*i+1][0] -= 2*numpy.pi
             while innov[2*i+1][0] < -numpy.pi:
                  innov[2*i+1][0] += 2*numpy.pi
        
        #print(len(sens.readings))
        #print("out loop")
        #print(innov)
       
        #Fill in the S matrix and the Kalmain gain matrix      
        S = numpy.dot(numpy.dot(sensmod_jac, covar_pred), numpy.transpose(sensmod_jac)) + sensmod_var
        K = numpy.dot(numpy.dot(covar_pred, numpy.transpose(sensmod_jac)), numpy.linalg.inv(S))
        
        #update the state and the covariance
        self.x = pred_state + numpy.dot(K,innov)
        self.P = covar_pred - numpy.dot(numpy.dot(K, sensmod_jac), covar_pred)
        #print(self.x)  
        
          
        #### ----- YOUR CODE GOES HERE ----- ####
    
    def sensor_callback(self,sens):

        # Publish state estimate 
        self.estimate(sens)
        est_msg = RobotPose()
        est_msg.header.stamp = sens.header.stamp
        est_msg.pose.x = self.x[0]
        est_msg.pose.y = self.x[1]
        est_msg.pose.theta = self.x[2]
        self.pub_est.publish(est_msg)

if __name__ == '__main__':
    rospy.init_node('state_estimator', anonymous=True)
    est = Estimator()
    rospy.spin()
