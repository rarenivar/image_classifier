package org.deeplearning4j.examples.animal_classifier;

import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.distribution.Distribution;

/**
 * Layers class containing static methods to return different types of layers for our neural network
 */
public class Layers {

    /**
     * Method to initialize and return the input convolution layer
     * @param layerName Convolution layer name
     * @param numberChannels Number of channels
     * @param numberFilters Number of filters
     * @param kernelSize Kernel size
     * @param stride Stride
     * @param pad Pad
     * @param biasValue Bias initial value
     * @return A ConvolutionLayer object initialize with the parameters passed
     */
    public static ConvolutionLayer initialConvolutionLayer(String layerName, int numberChannels,
                                                           int numberFilters, int[] kernelSize, int[] stride, int[] pad,
                                                           double biasValue) {
        ConvolutionLayer builder = new ConvolutionLayer.Builder(kernelSize, stride, pad)
            .name(layerName)
            .biasInit(biasValue)
            .nIn(numberChannels)
            .nOut(numberFilters)
            .build();
        return builder;
    }

    /**
     * Returns a Convolution layer with the parameters passed. The parameters function are explained by their names
     * @param layerName
     * @param numberFilters
     * @param stride
     * @param pad
     * @param kernelSize
     * @param biasValue
     * @return Returns a Convolution layer with the parameters passed
     */
    public static ConvolutionLayer convolutionLayer(String layerName, int numberFilters, int[] stride, int[] pad,
                                                    int[] kernelSize, double biasValue) {
        return new ConvolutionLayer.Builder(kernelSize, stride, pad)
            .name(layerName)
            .biasInit(biasValue)
            .nOut(numberFilters)
            .build();
    }

    /**
     * Returns a SubsamplingLayer
     * @param layerName
     * @param stride
     * @param kernel
     * @return Returns a SubsamplingLayer
     */
    public static SubsamplingLayer subsamplingLayer(String layerName, int[] stride, int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, stride)
            .name(layerName)
            .build();
    }

    /**
     * Returns a DenseLayer object
     * @param layerName
     * @param numberFilters
     * @param bias
     * @param dropOut
     * @param dist
     * @return Returns a DenseLayer object
     */
    public static DenseLayer denseLayer(String layerName, int numberFilters, double bias, double dropOut,
                                            Distribution dist) {
        return new DenseLayer.Builder()
            .name(layerName)
            .dist(dist)
            .dropOut(dropOut)
            .biasInit(bias)
            .nOut(numberFilters)
            .build();
    }
}
