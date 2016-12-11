package org.deeplearning4j.examples.animal_classifier;

import java.io.File;
import java.util.Random;
import java.util.Arrays;
import java.util.List;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class AnimalClassifier {

    protected static final Logger log = LoggerFactory.getLogger(AnimalClassifier.class);

    public void Execute() throws Exception {

        log.info("Load data....");
        /**cd
         * Data Setup -> organize and limit data file paths:
         *  - mainPath = path to image files
         *  - fileSplit = define basic dataset split with limits on format
         *  - pathFilter = define additional file load filter to limit size and balance batch content
         **/
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        File mainPath = new File(System.getProperty("user.dir"), "dl4j-examples/src/main/resources/animals");
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, Configuration.rng);
        BalancedPathFilter pathFilter = new BalancedPathFilter(Configuration.rng, labelMaker, Configuration.numImages, Configuration.numLabels, Configuration.batchSize);

        /**
         * Data Setup -> train test split
         *  - inputSplit = define train and test split
         **/
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, Configuration.numImages * (1 + Configuration.splitTrainTest), Configuration.numImages * (1 - Configuration.splitTrainTest));
        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];

        /**
         * Data Setup -> transformation
         *  - Transform = how to tranform images and generate large dataset to train on
         **/
       // put image transforms here

        /**
         * Data Setup -> normalization
         *  - how to normalize images and generate large dataset to train on
         **/
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

        log.info("Build model....");

        // Uncomment below to try AlexNet. Note change height and width to at least 100
//        MultiLayerNetwork network = new AlexNet(height, width, channels, numLabels, seed, iterations).init();


        MultiLayerNetwork network;
//        switch (modelType) {
//            case "LeNet":
//                network = lenetModel();
//                break;
//            case "AlexNet":
                network = MultiLayerNetworks.ImageNeuralNetwork(0.01, OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT, 1); // last parameter number of iterations, should be 1
//                break;
//            case "custom":
//                network = customModel();
//                break;
//            default:
//                throw new InvalidInputTypeException("Incorrect model provided.");
//        }
        network.init();
        network.setListeners(new ScoreIterationListener(1));

        /**
         * Data Setup -> define how to load data into net:
         *  - recordReader = the reader that loads and converts image data pass in inputSplit to initialize
         *  - dataIter = a generator that only loads one batch at a time into memory to save memory
         *  - trainIter = uses MultipleEpochsIterator to ensure model runs through the data for all epochs
         **/
        ImageRecordReader recordReader = new ImageRecordReader(Configuration.imageHeight, Configuration.imageWidth, Configuration.numChannels, labelMaker);
        DataSetIterator dataIter;
        MultipleEpochsIterator trainIter;


        // Train with transformations
        List<ImageTransform> listTransforms = imageTransforms();
        for (ImageTransform transform : listTransforms) {
            System.out.print("\nTraining on transformation: " + transform.getClass().toString() + "\n\n");
            recordReader.initialize(trainData, transform);
            dataIter = new RecordReaderDataSetIterator(recordReader, Configuration.batchSize, 1, Configuration.numLabels);
            scaler.fit(dataIter);
            dataIter.setPreProcessor(scaler);
            trainIter = new MultipleEpochsIterator(Configuration.epochNum, dataIter, Configuration.EpochsIteratorQueueSize);
            network.fit(trainIter);
        }

        log.info("Evaluate model....");
        recordReader.initialize(testData);
        dataIter = new RecordReaderDataSetIterator(recordReader, Configuration.batchSize, 1, Configuration.numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        Evaluation eval = network.evaluate(dataIter);
        log.info(eval.stats(true));

//        // Example on how to get predict results with trained model
//        dataIter.reset();
//        DataSet testDataSet = dataIter.next();
//        String expectedResult = testDataSet.getLabelName(0);
//        List<String> predict = network.predict(testDataSet);
//        String modelResult;
//        modelResult = predict.get(0);
//        System.out.print("\nFor a single example that is labeled " + expectedResult + " the model predicted " + modelResult + "\n\n");


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
        return Arrays.asList(new ImageTransform[]{flipTransform, warpTransform, flipTransform2, warpTransform2});
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
