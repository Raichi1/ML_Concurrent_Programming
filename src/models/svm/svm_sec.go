package main

import (
	"fmt"
)

type SVMModel struct {
	Weights []float64
	Bias    float64
}

func trainSVMSequential(data [][]float64, labels []float64) *SVMModel {
	weights := make([]float64, len(data[0]))
	bias := 0.0
	learningRate := 0.01
	epochs := 1000

	for epoch := 0; epoch < epochs; epoch++ {
		for i := range data {
			prediction := dotProduct(weights, data[i]) + bias
			error := labels[i] - prediction
			for j := range weights {
				weights[j] += learningRate * error * data[i][j]
			}
			bias += learningRate * error
		}
	}

	return &SVMModel{Weights: weights, Bias: bias}
}

func dotProduct(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func main() {
	data := [][]float64{{1, 2}, {2, 3}, {3, 4}}
	labels := []float64{1, 2, 3}
	model := trainSVMSequential(data, labels)
	fmt.Println("Trained SVM Model:", model)
}
