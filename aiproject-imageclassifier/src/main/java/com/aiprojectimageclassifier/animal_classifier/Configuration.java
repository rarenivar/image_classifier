package com.aiprojectimageclassifier.animal_classifier;

import java.util.Random;

/**
 * Class with static values for the configuration of the AnimalClassifier class
 */
public class Configuration {

    // Number of images contained in the training data set
    public static int numImages = 268;
    // Number of channels for the convolution layer
    public static int numChannels = 3;
    public static Random rng = new Random(100);
    // Batch size when training the data set
    public static int batchSize = 100;
    // Number of different labels of our image classification: frog, shark, cat and giraffe
    public static int numLabels = 4;
    // Queue size of the EpochsIterator object, which will be used when training our neural network
    public static int EpochsIteratorQueueSize = 2;
    // Dimension of the images when the neural network during the training process
    public static int imageHeight = 100;
    public static int imageWidth = 100;

}
