package org.deeplearning4j.examples.animal_classifier;

import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.distribution.Distribution;


public class Layers {

    public static ConvolutionLayer initialConvolutionLayer(String layerName, int numberChannels,
                                                           int numberFilters, int[] kernelSize, int[] stride, int[] pad,
                                                           double biasValue) {
        ConvolutionLayer builder = new ConvolutionLayer.Builder(kernelSize, stride, pad)
            // convolution layer name
            .name(layerName)
            // bias initial value
            .biasInit(biasValue)
            // number of channels
            .nIn(numberChannels)
            // number of filters (depth)
            .nOut(numberFilters)
            // returns the convolution layer object
            .build();
        return builder;
    }


    public static ConvolutionLayer convolutionLayer(String layerName, int numberFilters, int[] stride, int[] pad,
                                                    int[] kernelSize, double biasValue) {
        return new ConvolutionLayer.Builder(kernelSize, stride, pad)
            .name(layerName)
            .biasInit(biasValue)
            .nOut(numberFilters)
            .build();
    }

    public static SubsamplingLayer subsamplingLayer(String layerName, int[] stride, int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, stride)
            .name(layerName)
            .build();
    }

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
