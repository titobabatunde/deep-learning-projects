# Intro to Deep Learning 11-685 HW1P2
Click [here](https://www.kaggle.com/competitions/11-785-f23-hw2p2-classification/leaderboard) for classification kaggle link and [here](https://www.kaggle.com/competitions/11-785-f23-hw2p2-verification/leaderboard) for verification kaggle link.
<details open>
<summary> Information about Homework <i> (click on triangle to open) </i> </summary>

## Problem Statement
Learn to build a CNN-based architecture for face classification and verification

1) Face Classification
    * Closed set multiclass classification problem where the subjects in the <br />test set have also been seen in the training set
    * The embeddings for the subject be linearly separable from each other
    * A face classifier that can extract feature vectors from face images: 
        * Feature extractor: Train model to learn facial features (skin tone <br />
              , hair color, nose size, etc) from an image of a face and represent <br />
              them as fixed length features vectors (face embeddings).
        * Classification layer: The feature vector obtained at the end will <br />
               be passed through an MLP to classify it among N categories, and <br />
               use cross-entropy loss for optimization. The feature vectors <br />
               obtained after training can then be used for the verification.
        * For multi-class classifiction: the input to your system is a face's image <br />
         and your model needs to predic the ID of the face.
        * If the dataset contains M total images that belong to N different people <br />
             (where M>N), your model needs to produce "good" face embeddings
2) Face Verification
    * Determining whether two face images are of the same person, without <br />
        knowing how the person is
    * You are given an examplar of a category of data face) and an unknown <br />
         instance, and you must determine if the two are from the same class
    * Train a model to extract discriminative feature vectors from images, <br />
          which have the property that feature vectors from any two images that <br />
          belong to the same person are close than feature vectors derived from <br />
          images of two different persons
    * Given a pair of facial images, extract feature vectors from both and <br />
         compute their similarity. If similarity exceeds a threshold, we declare <br />
         a match.
    * A verification system that computes the similarity between feature
        vectors of two images. Here is a simple verification system:
        * Extracting feature vectors from image 1 -> fvector1
        * Extracting feature vectors from image 1 -> fvector2
        * fvector1, fvector2 -> similarity metric -> similarity score
        * In actuality, we compare each unknown identity to all the known
             identities and then decide whether this unknown identity matches any
             known identity using a threshold method, and predic the known
             identity with the highest similarity score
3) Data description
    * Dataset used is a subset of the VGGFace2 dataset. The images have been <br />
        resized to 224 x 224 pixels.
    * The classification dataset consists of 7,001 identities that are <br />
         balanced so that each class has the equal number of training images.
    * The verification dataset consists of 1080 identities that are split <br />
          into 360 for the Dev-set and 720 for the Test-set. Each of these <br />
          unknown identities will need to be mapped to either one of the 960 known <br />
          identities, or to n000000, the "no correspondence" label for the <br />
          remaining 120 identities.

4) Dataset Class - ImageFolder
    * We will be using the ImageFolder class from the torchvision library <br />
        and passing it the path to the training and validation dataset
    * Images in subfolder classification_data are arranged ina way that is <br />
        compatible with this dataset class.
    * The ImageFolder class will automatically infer the labels and make a <br />
        dataset object, which we can then pass on to the dataloader
    * Remember to also pass the image tranforms to the dataset class for doing <br />
      data augmentation
        * torchvision.transforms is designed for image transformations such as: <br />
             resizing, cropping, flipping, rotating, adjusting brightness/constrast, <br />
             normalizing pixel values, and converting images to tensors.
        * image transformations provide key advantages keep in minda that <br />
             these should reflect what you expect in real world ie. no vertical <br />
             flips)):
            * More data
            *  Preventing overfitting (reduce memorization of specific images)
            * Invariance (irrespective of orientation or position)
            * Better Generalization enhance performace on unseen data)

5) File Structure
    * Each sub-folder in train, dev, and test contains images of one person, <br />
        and the name of that sub-folder represents their ID
        * train: use the train set to train your model both for the <br />
            classification and verification task.
        * dev: use dev set to validate the classification accuracy.
        * test: assign IDs for images in the range of [0,7000] in test and <br />
              submit result. ImageFolder dataset by default maps each class to <br />
              such an ID.
    * classification_sample_submission.csv: this is a samle submission file for <br />
        classification where the first column is the image file names and the <br />
        second column is the image label.
    * For the verification dataset folder:
        * known: the directory of all 960 known identities.
        * unknown_test: the directory containing images for Verification Test. <br />
             720 images of unknown identites.
        * unknown_dev: the directory containing 360 images of uknown <br />
              identities, which you are given the ground truth mapping for.
        * verification_dev.csv: list of ground truth identity labels <br />
             (mapped to a known identity in folder or "no correspondence" label).
        * verification_sample_submission.csv: this is a sample submission file <br />
            for face verification. The first column is the index of the image <br />
            files and the second column is the label of each image.
    * classification acc = # correctly classified images/total images
    * verification acc =  # correctly match unknown identities/ total known identities

6) Create deeper layers with residual networks (resnets)
    * Skip connections allow us to take the activations of one layer and suddenly <br />
        feed it to another layer, even much deeper in the network.
    * Resnets are made of residual blocks, which are a set of layers that are <br />
        connected to each other, and the input of the first layer is added to the <br />
        output of the last layer in the block (residual connection). <br />
        There are several other blocks that make use of residual blocks and <br />
        connections: <br />
        MobilNet, ResNet, and ConvNet, ConvNeXt, etc (you can combine different blocks)

7) Similarity metric
    * For each unknown identity, you will have to predics the known identity that <br />
        is corresponds with (i.e. with highest similarity value) or, if the <br />
        similarity score is below the threshold, predic that it is not represented <br />
        in the known set.
8) Try other loss functions: center-loss, triplet-loss, pair-wise loss, LM, L_GM,...
    * pair-wise loss -> look into triplet loss

## Kaggle Cuttoffs
1. Classification:
    * HIGH - 90% accuracy
    * MEDIUM - 84% accuracy
    * LOW - 82% accuracy

2. Verification:
    * HIGH - 55% accuracy
    * MEDIUM - 49% accuracy
    * LOW - 30% accuracy
</details>

<br />
<br />

## Experimental Details
Click [here](https://wandb.ai/11685-cmu/hw2p2-ablations/table?workspace=user-titobabatunde) for wandb link <br />
1. Different architectures from start to finish with final classification accuracy of 87.5% <br /> and recognition accuracy of 51.2% before 60% of data was added. 
2. Shared Parameters:
    * Batch size of 128, CrossEntropyLoss, and an evalution verification threshold of 0.4. 
    * All architectures have a ConvBlock at the beginning (kernel=7, stride=2) with MaxPool <br /> (kernel=3, stride=2) at the beginning and AdaptiveAvgPool(1,1) at the end. 
    * Kaiming normal was used as an initializer for all Conv and Linear layers. 
    * Early stopping was used throughout
3. I would say that I didn't have as much time to play around with the verification <br />  portion (because not team mate collaboration) and spent longer on the classification <br /> architecture for extracting features. 
    * Therefore, logs only show accuracy for classification
4. Last row are the parameters for final architecture
| Architecture (+ channels for each blocks) | Epochs | <div style="width:90px">Dropout</div> | scheduler | optimizer | lr | Training Acc | Validation Acc | SE | activation | <div style="width:290px">Comments</div>|
| :---------------- | :------: | :----: | :------: | :----: | :------: | :----: | :------: | :----: | :----: | :------------------------ |
| ConvNeXt blocks, without deepwise, with transitional Conv blocks between, <br /> channels=[32,32,64,64,128,128,128,128] | 500 | False | CosineAnnealingWarmingRestarts(20) | SGD, nesterov | 0.01 |  91.8% | 69.8% | False | relu | Followed most of instructions from piazza, clearly overfit |
| InvBottleNeck blocks, without deepwise, with transitional Conv blocks between, <br /> channels=[32,64,64,128,128,256] | 500 | False | CosineAnnealingWarmingRestarts(20) | SGD, nesterov | 0.01 |  99.9% | 79.2% | False | relu | Validation Accuracy increased, decided I'm gonna stick with MobileNet Block |
| InvBottleNeck blocks, without deepwise, with transitional Conv blocks between, <br /> channels=[32,64,128,256] | 500 | 0.1 | ReduceLROnPlateau | AdamW | 0.01 |  99.9% | 69.3% | False | relu | Validation Accuracy dropped |
| InvBottleNeck blocks, with deepwise, with transitional Conv blocks between, <br /> channels=[32,64,128,256,512] | 500 | [0.2,0,0.2,0,0.2] | CosineAnnealingWarmingRestarts(5) | AdamW | 0.01 |  97.7% | 65.4% | [false,false,true,true,true] | relu6 | Maybe increase scheduler T and make architecture more sophisticated? |
| InvBottleNeck blocks, with deepwise, with transitional Conv blocks between, <br /> channels=[32,64,128,256,512],<br /> stride=[1,1,2,2,2] | 500 | [0.2,0,0.2,0,0.2] | CosineAnnealingWarmingRestarts(20) | SGD | 0.01 |  80.5% | 68.0% | [true,true,true,true,true] | relu6 | Okay so gap b/w validation and training is decreasing |
| InvBottleNeck blocks, with deepwise, with transitional Conv blocks between, <br /> channels=[32,64,128,256,512],<br /> stride=[1,1,2,2,2] | 150 | [0.2,0,0.2,0,0.2] | CosineAnnealingWarmingRestarts(20) | SGD | 0.1 |  17.6% | 17.4% | [true,true,true,true,true] | relu6 | not good... |
| InvBottleNeck blocks, with deepwise, with transitional Conv blocks between, <br /> channels=[32,64,128,256,512],<br /> stride=[1,1,2,2,2] | 150 | [0.2,0,0.2,0,0.2] | CosineAnnealingWarmingRestarts(50) | SGD | 0.01 |  73% | 63% | [true,true,true,true,true] | relu6 | not good... |
| InvBottleNeck blocks, with deepwise, with transitional Conv blocks between, <br /> channels=[32,64,128,256,512],<br /> stride=[1,1,2,2,2] | 150 | [0.2,0,0.2,0,0.2] | CosineAnnealingWarmingRestarts(60) | SGD | 0.01 |  71% | 60.5% | [true,true,true,true,true] | relu6 | maybe make scheduler wide? |
| InvBottleNeck blocks, with deepwise, with transitional Conv blocks between, <br /> channels=[32,64,128,256,512],<br /> stride=[1,1,2,2,2] | 95 | [0.2,0,0.2,0,0.2] | CosineAnnealingWarmingRestarts(100) | SGD | 0.01 |  100% | 86.5% | [true,true,true,true,true] | relu6 | play areound with scheduler width |
| InvBottleNeck blocks, with deepwise, with transitional Conv blocks between, <br /> channels=[32,64,128,256,512],<br /> stride=[1,1,2,2,2] | 145 | [0.2,0,0.2,0,0.2] | CosineAnnealingWarmingRestarts(150) | SGD | 0.01 |  100% | 87.1% | [true,true,true,true,true] | relu6 | play areound with scheduler width and epoch to end before restarts |
| InvBottleNeck blocks, with deepwise, with transitional Conv blocks between, <br /> channels=[32,64,128,256,512],<br /> stride=[1,1,2,2,2] | 195 | [0.2,0,0.2,0,0.2] | CosineAnnealingWarmingRestarts(200) | SGD | 0.01 |  100% | 87.3% | [true,true,true,true,true] | relu6 | play areound with scheduler width and epoch to end before restarts |
| InvBottleNeck blocks, with deepwise, with transitional Conv blocks between, <br /> channels=[32,64,128,256,512],<br /> stride=[1,1,2,2,2] | 478 | [0.2,0,0.2,0,0.2] | CosineAnnealingWarmingRestarts(20) | SGD | 0.001 |  100% | 87.4% | [true,true,true,true,true] | relu6 | play areound with scheduler width and epoch to end before restarts |


## Running Instructions
python3 main.py
