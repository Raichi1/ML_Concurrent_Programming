package main

import (
	"fmt"
	"math"
	"math/rand"
)

// Red Neuronal Artificial (ANN)
type ANN struct {
	weights                           [][]float64
	bias                              []float64
	inputSize, hiddenSize, outputSize int
}

// Inicializa la red neuronal
func newANN(inputSize, hiddenSize, outputSize int) *ANN {
	weights := make([][]float64, inputSize+hiddenSize)
	for i := range weights {
		weights[i] = make([]float64, hiddenSize+outputSize)
		for j := range weights[i] {
			weights[i][j] = rand.Float64()*2 - 1
		}
	}
	bias := make([]float64, hiddenSize+outputSize)
	for i := range bias {
		bias[i] = rand.Float64()*2 - 1
	}
	return &ANN{
		weights:    weights,
		bias:       bias,
		inputSize:  inputSize,
		hiddenSize: hiddenSize,
		outputSize: outputSize,
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
func (ann *ANN) forward(inputs []float64) []float64 {
	hiddenLayer := make([]float64, ann.hiddenSize)
	outputLayer := make([]float64, ann.outputSize)

	// Cálculo de la capa oculta
	for i := range hiddenLayer {
		sum := ann.bias[i]
		for j := range inputs {
			sum += inputs[j] * ann.weights[j][i]
		}
		hiddenLayer[i] = sigmoid(sum)
	}

	// Cálculo de la capa de salida
	for i := range outputLayer {
		sum := ann.bias[ann.hiddenSize+i]
		for j := range hiddenLayer {
			sum += hiddenLayer[j] * ann.weights[ann.inputSize+j][i]
		}
		outputLayer[i] = sigmoid(sum)
	}

	return outputLayer
}

// Función de error cuadrático medio
func meanSquaredError(yTrue, yPred []float64) float64 {
	sum := 0.0
	for i := range yTrue {
		diff := yTrue[i] - yPred[i]
		sum += diff * diff
	}
	return sum / float64(len(yTrue))
}

// Ajuste de pesos y sesgos (Backpropagation)
func (ann *ANN) backpropagate(inputs []float64, labels []float64, output []float64, learningRate float64) {
	// Calcular el error de la capa de salida
	outputError := make([]float64, ann.outputSize)
	for i := range outputError {
		outputError[i] = labels[i] - output[i]
	}

	// Calcular el error de la capa oculta
	hiddenLayerError := make([]float64, ann.hiddenSize)
	for i := range hiddenLayerError {
		sum := 0.0
		for j := range outputError {
			sum += outputError[j] * ann.weights[ann.inputSize+i][j]
		}
		hiddenLayerError[i] = sigmoidDeriv(sum) * sum
	}

	// Actualizar pesos y sesgos
	for i := 0; i < ann.inputSize; i++ {
		for j := 0; j < ann.hiddenSize; j++ {
			ann.weights[i][j] += learningRate * hiddenLayerError[j] * inputs[i]
		}
	}
	for i := 0; i < ann.hiddenSize; i++ {
		for j := 0; j < ann.outputSize; j++ {
			ann.weights[ann.inputSize+i][j] += learningRate * outputError[j] * sigmoidDeriv(hiddenLayerError[i])
		}
	}
	for i := range ann.bias {
		if i < ann.hiddenSize {
			ann.bias[i] += learningRate * hiddenLayerError[i]
		} else {
			ann.bias[i] += learningRate * outputError[i-ann.hiddenSize]
		}
	}
}

// Entrenamiento de la red neuronal
func (ann *ANN) train(data [][]float64, labels [][]float64, epochs int, learningRate float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		for i := range data {
			output := ann.forward(data[i])
			ann.backpropagate(data[i], labels[i], output, learningRate)
		}
	}
}

// Métricas de evaluación
func evaluate(predictions, labels [][]float64) (float64, float64, float64) {
	var tp, fp, fn, tn float64
	for i := range predictions {
		pred := predictions[i][0] > 0.5
		trueLabel := labels[i][0] > 0.5

		if pred && trueLabel {
			tp++
		} else if pred && !trueLabel {
			fp++
		} else if !pred && trueLabel {
			fn++
		} else if !pred && !trueLabel {
			tn++
		}
	}

	precision := tp / (tp + fp)
	recall := tp / (tp + fn)
	f1 := 2 * (precision * recall) / (precision + recall)

	return precision, recall, f1
}

func main() {
	rand.Seed(42)

	// Definir dimensiones
	inputSize := 3
	hiddenSize := 5
	outputSize := 1

	// Crear red neuronal
	ann := newANN(inputSize, hiddenSize, outputSize)

	// Datos de ejemplo
	data := [][]float64{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}}
	labels := [][]float64{{0.0}, {1.0}}

	// Entrenar red neuronal
	ann.train(data, labels, 100, 0.01)

	// Hacer predicciones
	predictions := make([][]float64, len(data))
	for i, d := range data {
		predictions[i] = ann.forward(d)
		// fmt.Printf("Prediction for %v: %v\n", d, predictions[i])
	}

	// Evaluar el modelo
	precision, recall, f1 := evaluate(predictions, labels)
	fmt.Printf("Precision: %v\n", precision)
	fmt.Printf("Recall: %v\n", recall)
	fmt.Printf("F1 Score: %v\n", f1)
}
