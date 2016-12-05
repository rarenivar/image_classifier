package org.deeplearning4j.examples.animal_classifier;

import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;

public class Layers {

    public static ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad)
            // convolution layer name
            .name(name)
            // number of channels
            .nIn(in)
            // number of filters (depth)
            .nOut(out)
            // bias initial value
            .biasInit(bias)
            // returns the convolution layer object
            .build();
    }

    public static ConvolutionLayer conv3x3(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {1,1})
            .name(name)
            .nOut(out)
            .biasInit(bias)
            .build();
    }

    public static ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(new int[]{5,5}, stride, pad)
            .name(name)
            .nOut(out)
            .biasInit(bias)
            .build();
    }

    public static SubsamplingLayer maxPool(String name, int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2,2})
            .name(name)
            .build();
    }

    public static DenseLayer fullyConnected(String name, int out, double bias, double dropOut, Distribution dist) {
        return new DenseLayer.Builder()
            .name(name)
            .nOut(out)
            .biasInit(bias)
            .dropOut(dropOut)
            .dist(dist)
            .build();
    }

}
