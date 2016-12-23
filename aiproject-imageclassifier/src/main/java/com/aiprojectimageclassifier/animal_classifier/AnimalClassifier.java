package com.aiprojectimageclassifier.animal_classifier;

import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.util.Arrays;
import java.util.List;
import java.util.Date;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.WarpImageTransform;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;

/**
 * An AnimalClassifier object’s responsibility is to scan, manage and process the training and test data before a deep
 * neural network can use it. In this particular case, the data will consist of a number of animal images. After this,
 * the AnimalClassifier initializes a MultiLayerNetwork object, which is an implementation of a deep neural network.
 * Additionally, this object creates the different image transforms that will be when the deep neural network starts
 * learning with the training data. Finally, once the deep neural network is done learning, this object will also test
 * the learning model with the previously defined testing data.
 */
public class AnimalClassifier {

    /**
     * Start executing the Classifier
     * @throws Exception
     */
    public void Execute() throws Exception {

        // Number of Epochs and Iterations, Learning Rates
        int[] epochValues = new int[] {5, 15, 25};
        double[] learningRateValues = new double[] {0.1, 0.01, 0.001};
        int[] iterationValues = new int[] {1, 2, 4};

        for (int i = 0; i < epochValues.length; i++) {
            for (int j = 0; j < iterationValues.length; j++) {
                for (int k = 0; k < learningRateValues.length; k++) {
                    System.out.println("=== Beginning of Animal Classifier execution ===");
                    Date startDate = new Date();
                    System.out.println("Start time: " + printCurrentDateAndTime(startDate));
                    System.out.println("Epoch value = " + epochValues[i]);
                    System.out.println("Iterations value = " + iterationValues[j]);
                    System.out.println("Learning Rate value = " + learningRateValues[k]);

                    // Object responsible for loading a batch into memory
                    DataSetIterator dataSetIterator;
                    // Iterates through the data on each Epoch on each learning model cycle
                    MultipleEpochsIterator multipleEpochsIterator;

                    ParentPathLabelGenerator parentPathLabelGenerator = new ParentPathLabelGenerator();
                    // Giving the images location, get the training and testing data in an InputSplit object
                    InputSplit trainInputSplit = getTrainingData("user.dir",
                        "aiproject-imageclassifier/src/main/resources/animals",
                        parentPathLabelGenerator);
                    InputSplit testInputSplit = getTestData("user.dir",
                        "aiproject-imageclassifier/src/main/resources/animals",
                        parentPathLabelGenerator);

                    // Normalizing images for training and testing
                    DataNormalization dataNormalization = new ImagePreProcessingScaler(0, 1);

                    MultiLayerNetwork deepNeuralNetwork;
                    deepNeuralNetwork = MultiLayerNetworks.ImageNeuralNetwork(learningRateValues[k],
                        OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT, iterationValues[j]);

                    // Initializing the deepNeuralNetwork
                    deepNeuralNetwork.init();

                    // An ImageRecordReader object is responsible for loading the image data with the appropriate
                    // configuration passed as arguments
                    ImageRecordReader imageRecordReader = new ImageRecordReader(Configuration.imageHeight,
                        Configuration.imageWidth, Configuration.numChannels, parentPathLabelGenerator);

                    // Train with transformations
                    List<ImageTransform> listTransforms = imageTransforms();
                    System.out.println("Initializing deep neural network training with transformation");
                    for (ImageTransform transform : listTransforms) {
                        System.out.println("Training... transformation used: " + transform.getClass().toString());
                        imageRecordReader.initialize(trainInputSplit, transform);
                        dataSetIterator = new RecordReaderDataSetIterator(imageRecordReader, Configuration.batchSize, 1,
                            Configuration.numLabels);
                        dataNormalization.fit(dataSetIterator);
                        dataSetIterator.setPreProcessor(dataNormalization);
                        multipleEpochsIterator = new MultipleEpochsIterator(epochValues[i], dataSetIterator,
                            Configuration.EpochsIteratorQueueSize);
                        deepNeuralNetwork.fit(multipleEpochsIterator);
                    }
                    Date endDate = new Date();
                    System.out.println("End time: " + endDate.toString());
                    // Calculating total times in seconds
                    long totalTime = (endDate.getTime() - startDate.getTime()) / 1000;
                    System.out.println("Configuration ... Epochs: " + epochValues[i] + " Iterations: " +
                        iterationValues[j] + " Learning Rate: " + learningRateValues[k]);
                    System.out.println("Neural Network training total times in seconds: " + totalTime);
                    // Testing the performance of the neural network - deepNeuralNetwork
                    AnalyzingDeepNeuralNetwork(imageRecordReader, testInputSplit, dataNormalization, deepNeuralNetwork);
                }
            }
        } // outer for loop
    }

    /**
     * Given the directory where the image data is, this returns an InputSplit object that contains the training data
     * ready to be used by a deep neural network
     * @param parentDirectory Root directory of the image data
     * @param dataLocation Child location of the image data
     * @return An InputSplit object containing the training data
     */
    public static InputSplit getTrainingData(String parentDirectory, String dataLocation,
                                             ParentPathLabelGenerator labelGenerator) {
        File filePath = new File(System.getProperty(parentDirectory), dataLocation);
        FileSplit fileSplit = new FileSplit(filePath, NativeImageLoader.ALLOWED_FORMATS, new Random(100));
        BalancedPathFilter balancedPathFilter = new BalancedPathFilter(Configuration.rng, labelGenerator,
            Configuration.numImages, Configuration.numLabels, Configuration.batchSize);
        InputSplit[] inputSplit = fileSplit.sample(balancedPathFilter,
            Configuration.numImages * (1.8),
            Configuration.numImages * (.2));
        return inputSplit[0];
    }

    /**
     * Given the directory where the image data is, this returns an InputSplit object that contains the testing data
     * ready to be used by a deep neural network
     * @param parentDirectory Root directory of the image data
     * @param dataLocation Child location of the image data
     * @return An InputSplit object containing the testing data
     */
    public static InputSplit getTestData(String parentDirectory, String dataLocation,
                                         ParentPathLabelGenerator labelGenerator) {
        File filePath = new File(System.getProperty(parentDirectory), dataLocation);
        FileSplit fileSplit = new FileSplit(filePath, NativeImageLoader.ALLOWED_FORMATS, new Random(100));
        BalancedPathFilter balancedPathFilter = new BalancedPathFilter(Configuration.rng, labelGenerator,
            Configuration.numImages, Configuration.numLabels, Configuration.batchSize);
        InputSplit[] inputSplit = fileSplit.sample(balancedPathFilter,
            Configuration.numImages * (1.8),
            Configuration.numImages * (.2));
        return inputSplit[1];
    }

    /**
     * After training the deep neural network, running this function will analyze the deep neural network and using
     * the DeepLearning Evaluation object, will return the following metrics: accuracy, recall, precision and F1 score
     * @param imageRecordReader
     * @param testInputSplit
     * @param dataNormalization
     * @param multiLayerNetwork
     */
    public static void AnalyzingDeepNeuralNetwork(ImageRecordReader imageRecordReader, InputSplit testInputSplit,
                                                  DataNormalization dataNormalization,
                                                  MultiLayerNetwork multiLayerNetwork) {
        System.out.println("=== Analyzing deep neural network ===");
        try {
            imageRecordReader.initialize(testInputSplit);
        } catch (IOException e) {
            System.out.println("Error in AnalyzingDeepNeuralNetwork, message: " + e.getMessage());
        }
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(imageRecordReader, Configuration.batchSize, 1,
            Configuration.numLabels);
        dataNormalization.fit(dataSetIterator);
        dataSetIterator.setPreProcessor(dataNormalization);
        Evaluation evaluation = multiLayerNetwork.evaluate(dataSetIterator);
        System.out.println(evaluation.stats(true));
    }

    /**
     * Static method that creates four different ImageTransforms objects, two FlipTransforms and two WarpImageTransforms
     * objects, which serve to artifically increase the data set by transforming our image data set, either
     * deterministacally or randomly.
     * @return A list of ImageTransform objects
     */
    public static List<ImageTransform> imageTransforms() {
        ImageTransform flipTransform = new FlipImageTransform(new Random(100));
        ImageTransform flipTransform2 = new FlipImageTransform(new Random(100));
        ImageTransform warpTransform = new WarpImageTransform(new Random(100), 50);
        ImageTransform warpTransform2 = new WarpImageTransform(new Random(100), 25);
        return Arrays.asList(new ImageTransform[]{
            flipTransform,
            warpTransform,
            flipTransform2,
            warpTransform2
        });
    }

    public static String printCurrentDateAndTime(Date date) {
        return date.toString();
    }

    /**
     * Point of execution
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        new AnimalClassifier().Execute();
    }
}
