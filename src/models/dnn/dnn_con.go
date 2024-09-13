package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
)

// Estructura del modelo DNN
type DNN struct {
	weights    [][][]float64
	biases     [][]float64
	layerSizes []int
	mu         sync.Mutex
}

// Configuración de la red neuronal
type neuralNetConfig struct {
	inputNeurons  int
	outputNeurons int
	hiddenNeurons int
	numEpochs     int
	learningRate  float64
}

// Inicializa el modelo DNN
func newDNN(layerSizes []int) *DNN {
	numLayers := len(layerSizes)
	weights := make([][][]float64, numLayers-1)
	biases := make([][]float64, numLayers-1)

	for i := 0; i < numLayers-1; i++ {
		weights[i] = make([][]float64, layerSizes[i])
		biases[i] = make([]float64, layerSizes[i+1])
		for j := 0; j < layerSizes[i]; j++ {
			weights[i][j] = make([]float64, layerSizes[i+1])
			for k := 0; k < layerSizes[i+1]; k++ {
				weights[i][j][k] = rand.Float64()*2 - 1
			}
		}
		for k := 0; k < layerSizes[i+1]; k++ {
			biases[i][k] = rand.Float64()*2 - 1
		}
	}
	return &DNN{
		weights:    weights,
		biases:     biases,
		layerSizes: layerSizes,
	}
}

// Función de activación Sigmoid
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Derivada de la función de activación Sigmoid
func sigmoidDeriv(x float64) float64 {
	sig := sigmoid(x)
	return sig * (1 - sig)
}

// Propagación hacia adelante
func (dnn *DNN) forward(inputs []float64) []float64 {
	layerInput := inputs
	for i := 0; i < len(dnn.weights); i++ {
		layerOutput := make([]float64, dnn.layerSizes[i+1])
		for j := 0; j < dnn.layerSizes[i+1]; j++ {
			sum := dnn.biases[i][j]
			for k := 0; k < dnn.layerSizes[i]; k++ {
				sum += layerInput[k] * dnn.weights[i][k][j]
			}
			layerOutput[j] = sigmoid(sum)
		}
		layerInput = layerOutput
	}
	return layerInput
}

// Ajuste de pesos y sesgos (Backpropagation)
func (dnn *DNN) backpropagate(inputs []float64, labels []float64, output []float64, learningRate float64) {
	deltas := make([][]float64, len(dnn.weights))
	for i := range deltas {
		deltas[i] = make([]float64, dnn.layerSizes[i+1])
	}

	// Calcular el error de la capa de salida
	outputError := make([]float64, dnn.layerSizes[len(dnn.layerSizes)-1])
	for i := range outputError {
		outputError[i] = labels[i] - output[i]
	}
	deltas[len(deltas)-1] = outputError

	// Calcular el error de las capas ocultas
	for i := len(dnn.weights) - 1; i >= 0; i-- {
		layerDelta := make([]float64, dnn.layerSizes[i])
		for j := range layerDelta {
			sum := 0.0
			for k := range deltas[i] {
				sum += deltas[i][k] * dnn.weights[i][j][k]
			}
			layerDelta[j] = sum * sigmoidDeriv(sum)
		}
		if i > 0 {
			deltas[i-1] = layerDelta
		}

		// Actualizar pesos y sesgos
		dnn.mu.Lock()
		for j := range dnn.weights[i] {
			for k := range dnn.weights[i][j] {
				dnn.weights[i][j][k] += learningRate * deltas[i][k] * inputs[j]
			}
		}
		for j := range dnn.biases[i] {
			dnn.biases[i][j] += learningRate * deltas[i][j]
		}
		dnn.mu.Unlock()

		// Actualizar inputs para la capa anterior
		if i > 0 {
			newInputs := make([]float64, dnn.layerSizes[i])
			for j := range newInputs {
				sum := dnn.biases[i-1][j]
				for k := range inputs {
					sum += inputs[k] * dnn.weights[i-1][k][j]
				}
				newInputs[j] = sigmoid(sum)
			}
			inputs = newInputs
		}
	}
}

// Entrenamiento de la red neuronal profunda concurrentemente
func (dnn *DNN) trainConcurrent(data [][]float64, labels [][]float64, epochs int, learningRate float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		var wg sync.WaitGroup
		for i := range data {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				output := dnn.forward(data[i])
				dnn.backpropagate(data[i], labels[i], output, learningRate)
			}(i)
		}
		wg.Wait()
	}
}

func main() {
	// Datos de ejemplo
	inputs := [][]float64{
		{0.1, 0.2, 0.3, 0.4},
		{0.5, 0.6, 0.7, 0.8},
		{0.9, 1.0, 1.1, 1.2},
	}
	labels := [][]float64{
		{0.0},
		{1.0},
		{0.0},
	}

	// Definir la arquitectura de la red y los parámetros de aprendizaje.
	config := neuralNetConfig{
		inputNeurons:  4,
		outputNeurons: 1,
		hiddenNeurons: 5,
		numEpochs:     1000,
		learningRate:  0.01,
	}

	// Crear la red neuronal profunda.
	dnn := newDNN([]int{config.inputNeurons, config.hiddenNeurons, config.outputNeurons})

	// Entrenar la red neuronal profunda concurrentemente.
	dnn.trainConcurrent(inputs, labels, config.numEpochs, config.learningRate)

	// Hacer predicciones y evaluar el modelo.
	for _, input := range inputs {
		output := dnn.forward(input)
		fmt.Printf("Input: %v, Output: %v\n", input, output)
	}
}
