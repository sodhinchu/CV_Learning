# Project Details
    Pick your last code
    Make sure  to Add CutOut to your code. It should come from your transformations (albumentations)
    Use this repo: https://github.com/davidtvs/pytorch-lr-finder (Links to an external site.) 
        Move LR Finder code to your modules
        Implement LR Finder (for SGD, not for ADAM)
        Implement ReduceLROnPlatea: https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau (Links to an external site.)
    Find best LR to train your model
    Use SDG with Momentum
    Train for 50 Epochs. 
    Show Training and Test Accuracy curves
    Target 88% Accuracy.
    Run GradCAM on the any 25 misclassified images. Make sure you mention what is the prediction and what was the ground truth label.
    Submit
    
Below images are the misclassification images and visualized using GradCam Visualizer

![image1](MisClassified_Images/bird_plane_1.png)
![image2](MisClassified_Images/car_truck_1.png)
![image3](MisClassified_Images/truck_car_1.png)
![image4](MisClassified_Images/cat_car_1.png)
![image5](MisClassified_Images/cat_dog_1.png)
![image6](MisClassified_Images/dog_cat_1.png)
![image7](MisClassified_Images/car_truck_4.png)
![image8](MisClassified_Images/cat_ship_1.png)
![image9](MisClassified_Images/cat_bird_1.png)
![image10](MisClassified_Images/deer_bird_1.png)
![image11](MisClassified_Images/deer_cat_1.png)
![image12](MisClassified_Images/dog_bird_1.png)
![image13](MisClassified_Images/horse_bird_1.png)
![image14](MisClassified_Images/horse_cat_1.png)
![image15](MisClassified_Images/horse_deer_1.png)
![image16](MisClassified_Images/horse_plane_1.png)
![image17](MisClassified_Images/plane_ship_1.png)
![image18](MisClassified_Images/ship_plane_1.png)
![image19](MisClassified_Images/ship_truck_1.png)
![image20](MisClassified_Images/car_truck_2.png)
![image21](MisClassified_Images/car_truck_3.png)
![image22](MisClassified_Images/deer_cat_2.png)
![image23](MisClassified_Images/dog_cat_2.png)
![image24](MisClassified_Images/dog_cat_3.png)
![image25](MisClassified_Images/dog_cat_4.png)

