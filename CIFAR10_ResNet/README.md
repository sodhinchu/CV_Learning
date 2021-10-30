# Project Work

Write a code which uses this new ResNet Architecture for Cifar10: <br /><br />
        PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]<br /><br />
        Layer1 -<br />
            X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]<br />
            R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] <br />
            Add(X, R1)<br /><br />
        Layer 2 -<br />
            Conv 3x3 [256k]<br />
            MaxPooling2D<br />
            BN<br />
            ReLU<br /><br />
        Layer 3 -<br />
            X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]<br />
            R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]<br />
            Add(X, R2)<br /><br />
        MaxPooling with Kernel Size 4<br />
        FC Layer <br />
        SoftMax<br /><br />
    Uses One Cycle Policy such that:<br />
        Total Epochs = 24<br />
        Max at Epoch = 5<br />
        LRMIN = FIND<br />
        LRMAX = FIND<br />
        NO Annihilation<br /><br />
    Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)<br />
    Batch size = 512<br /><br />
    Target Accuracy: 90%. <br />
