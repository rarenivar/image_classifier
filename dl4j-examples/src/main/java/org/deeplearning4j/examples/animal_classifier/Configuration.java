package org.deeplearning4j.examples.animal_classifier;


import org.deeplearning4j.examples.convolution.AnimalsClassification;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.Random;

public class Configuration {

    public static int iterations = 1;
    public static int epochNum = 50;//100; //50;
    public static int numImages = 400; //80;
    public static int numChannels = 3;
    public static long seedNumber= 42;
    public static Random rng = new Random(seedNumber);

    public static int batchSize = 20;
    public static int numLabels = 4;

    public static int listenerFreq = 1;
    public static String modelType = "AlexNet"; // LeNet, AlexNet or Custom but you need to fill it out

    public static int nCores = 2;
    public static double splitTrainTest = 0.8;
    public static int imageHeight = 100;
    public static int imageWidth = 100;

}
