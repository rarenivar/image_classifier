package org.deeplearning4j.examples.animal_classifier;

import org.deeplearning4j.examples.convolution.AnimalsClassification;

import java.util.Random;

public class Main {

    protected static int width = 100;
    protected static int channels = 3;
    protected static int numExamples = 80;
    protected static int numLabels = 4;
    protected static int batchSize = 20;

    protected static long seed = 42;
    protected static Random rng = new Random(seed);
    protected static int listenerFreq = 1;
    protected static int iterations = 1;
    protected static int epochs = 50;
    protected static double splitTrainTest = 0.8;
    protected static int nCores = 2;
    protected static boolean save = false;

    public static void main(String[] args) throws Exception {
    }
}
