package org.deeplearning4j.examples.animal_classifier;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.LocalResponseNormalization;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;


public class MultiLayerNetworks {

    public static MultiLayerNetwork ImageNeuralNetwork(double learningRate, OptimizationAlgorithm optimizationAlgorithm,
                                                       int theIterations) {
        /**
         * AlexNet model interpretation based on the original paper ImageNet Classification with Deep Convolutional Neural Networks
         * and the imagenetExample code referenced.
         * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
         **/


        MultiLayerConfiguration multiLayerConfiguration = new NeuralNetConfiguration.Builder()
            .seed(Configuration.seedNumber)
            .weightInit(WeightInit.DISTRIBUTION)
            .dist(new NormalDistribution(0.0, 0.01))
            .activation("relu")
            .updater(Updater.NESTEROVS)
            .iterations(theIterations)
            .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
            .optimizationAlgo(optimizationAlgorithm)  //OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT
            .learningRate(learningRate) // 1e-2
            .biasLearningRate(1e-2*2)
            .learningRateDecayPolicy(LearningRatePolicy.Step)
            .lrPolicyDecayRate(0.1)
            .lrPolicySteps(100000)
            .regularization(true)
            .l2(5 * 1e-4)
            .momentum(0.9)
            .miniBatch(false)
            .list()
            .layer(0, Layers.convolutionLayer("cnn1", Configuration.numChannels, 96, new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3}, 0))
            .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
            .layer(2, Layers.maxPool("maxpool1", new int[]{3,3}))
            .layer(3, Layers.conv5x5("cnn2", 256, new int[] {1,1}, new int[] {2,2}, 1))
            .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
            .layer(5, Layers.maxPool("maxpool2", new int[]{3,3}))
            .layer(6, Layers.conv3x3("cnn3", 384, 0))
            .layer(7, Layers.conv3x3("cnn4", 384, 1))
            .layer(8, Layers.conv3x3("cnn5", 256, 1))
            .layer(9, Layers.maxPool("maxpool3", new int[]{3,3}))
            .layer(10, Layers.fullyConnected("ffn1", 4096, 1, 0.5, new GaussianDistribution(0, 0.005)))
            .layer(11, Layers.fullyConnected("ffn2", 4096, 1, 0.5, new GaussianDistribution(0, 0.005)))
            .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .name("output")
                .nOut(Configuration.numLabels)
                .activation("softmax")
                .build())
            .backprop(true)
            .pretrain(false)
            .cnnInputSize(Configuration.imageHeight, Configuration.imageWidth, Configuration.numChannels)
            .build();


//        MultiLayerConfiguration multiLayerConfiguration = new NeuralNetConfiguration.Builder()
//            .seed(Configuration.seedNumber)
//            .weightInit(WeightInit.DISTRIBUTION)
//            .dist(new NormalDistribution(0.0, 0.01))
//            .activation("relu")
//            .updater(Updater.NESTEROVS)
//            .iterations(iterations)
//            .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
//            .optimizationAlgo(optimizationAlgorithm)  //OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT
//            .learningRate(learningRate) // 1e-2
//            .biasLearningRate(1e-2*2)
//            .learningRateDecayPolicy(LearningRatePolicy.Step)
//            .lrPolicyDecayRate(0.1)
//            .lrPolicySteps(100000)
//            .regularization(true)
//            .l2(5 * 1e-4)
//            .momentum(0.9)
//            .miniBatch(false)
//            .list()
//            .layer(0, Layers.convInit("cnn1", Configuration.numChannels, 96, new int[]{11, 11}, new int[]{4, 4}, new int[]{3, 3}, 0))
//            .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
//            .layer(2, Layers.maxPool("maxpool1", new int[]{3,3}))
//            .layer(3, Layers.conv5x5("cnn2", 256, new int[] {1,1}, new int[] {2,2}, 1))
//            .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
//            .layer(5, Layers.maxPool("maxpool2", new int[]{3,3}))
//            .layer(6, Layers.conv3x3("cnn3", 384, 0))
//            // .layer(7, Layers.conv3x3("cnn4", 384, 1))
//            // .layer(8, Layers.conv3x3("cnn5", 256, 1))
//            .layer(7, Layers.maxPool("maxpool3", new int[]{3,3}))
//            .layer(8, Layers.fullyConnected("ffn1", 4096, 1, 0.5, new GaussianDistribution(0, 0.005)))
//            //  .layer(11, Layers.fullyConnected("ffn2", 4096, 1, 0.5, new GaussianDistribution(0, 0.005)))
//            .layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                .name("output")
//                .nOut(Configuration.numLabels)
//                .activation("softmax")
//                .build())
//            .backprop(true)
//            .pretrain(false)
//            .cnnInputSize(Configuration.imageHeight, Configuration.imageWidth, Configuration.numChannels)
//            .build();



        return new MultiLayerNetwork(multiLayerConfiguration);

    }

}
