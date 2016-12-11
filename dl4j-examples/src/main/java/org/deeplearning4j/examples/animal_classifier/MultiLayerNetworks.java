package org.deeplearning4j.examples.animal_classifier;

import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.conf.layers.LocalResponseNormalization;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * This layer is responsible for initializing and setting up the configuration of our deep neural network.
 */
public class MultiLayerNetworks {

    /**
     * The MultiLayerNetwork object returned from this static method is based on the research paper
     * "ImageNet Classification with Deep Convolutional Neural Networks", which can be found at
     * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
     * This will be used by the AnimalClassifier class to create different deep neural networks with different
     * configurations based on the three parameters passed
     * @param learningRate double value representing the learning rate our neural network will use
     * @param optimizationAlgorithm An OptimizationAlgorithm object which reflects the algoritm the MultiLayerNetwork
     *                              object will use
     * @param theIterations int value representing the number of times the learning model will iterate through the data
     * @return A MultiLayerNetwork object with the appropriate configuration
     */
    public static MultiLayerNetwork ImageNeuralNetwork(double learningRate, OptimizationAlgorithm optimizationAlgorithm,
                                                       int theIterations) {
        MultiLayerConfiguration multiLayerConfiguration = new NeuralNetConfiguration.Builder()
            .iterations(theIterations)
            .lrPolicyDecayRate(0.1)
            .lrPolicySteps(100000)
            .learningRateDecayPolicy(LearningRatePolicy.Step)
            .dist(new NormalDistribution(0.0, 0.01))
            .weightInit(WeightInit.DISTRIBUTION)
            .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
            .optimizationAlgo(optimizationAlgorithm)  //OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT
            .l2(0.0005) // 5 * 1e - 4
            .seed(100)
            .momentum(0.9)
            .learningRate(learningRate) // 1e-2
            .biasLearningRate(0.02) // 1e-2*2
            .regularization(true)
            .activation("relu")
            .updater(Updater.NESTEROVS)
            .miniBatch(false)
            .list()
            .layer(0, Layers.initialConvolutionLayer("initialConvolutionLayer", Configuration.numChannels, 96,
                new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3}, 0))
            .layer(1, new LocalResponseNormalization.Builder().name("firstResponseNormalization").build())
            .layer(2, Layers.subsamplingLayer("firstMaxpoolLayer", new int[]{2,2}, new int[]{3,3}))
            .layer(3, Layers.convolutionLayer("secondConvolutionLayer", 256, new int[] {1,1}, new int[] {2,2},
                new int[] {5,5}, 1))
            .layer(4, new LocalResponseNormalization.Builder().name("secondResponseNormalization").build())
            .layer(5, Layers.subsamplingLayer("thirdMaxpoolLayer", new int[]{2,2}, new int[]{3,3}))
            .layer(6, Layers.convolutionLayer("fourthConvolutionLayer", 384, new int[] {1,1},
                new int[] {2,2}, new int[] {3,3}, 0))
            .layer(7, Layers.convolutionLayer("fifthConvolutionLayer", 384, new int[] {1,1},
                new int[] {2,2}, new int[] {3,3}, 1))
            .layer(8, Layers.convolutionLayer("sixthConvolutionLayer", 256, new int[] {1,1},
                new int[] {2,2}, new int[] {3,3}, 1))
            .layer(9, Layers.subsamplingLayer("secondMaxpoolLayer", new int[]{2,2}, new int[]{3,3}))
            .layer(10, Layers.denseLayer("firstFullyConnectedLayer", 4096, 1, 0.5, new GaussianDistribution(0, 0.005)))
            .layer(11, Layers.denseLayer("secondFullyConnectedLayer", 4096, 1, 0.5, new GaussianDistribution(0, 0.005)))
            .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .name("outpuLayer")
                .nOut(Configuration.numLabels)
                .activation("softmax")
                .build())
            .pretrain(false)
            .backprop(true)
            .cnnInputSize(Configuration.imageHeight, Configuration.imageWidth, Configuration.numChannels)
            .build();
        return new MultiLayerNetwork(multiLayerConfiguration);
    }
}
