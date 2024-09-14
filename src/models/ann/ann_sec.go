package ann

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// Red Neuronal Artificial (ANN)
type ANN struct {
	weights1, weights2                [][]float64
	bias1, bias2                      []float64
	inputSize, hiddenSize, outputSize int
}

// Inicializa la red neuronal
func newANN(inputSize, hiddenSize, outputSize int) *ANN {
	weights1 := make([][]float64, inputSize)
	for i := range weights1 {
		weights1[i] = make([]float64, hiddenSize)
		for j := range weights1[i] {
			weights1[i][j] = rand.Float64()*2 - 1
		}
	}

	weights2 := make([][]float64, hiddenSize)
	for i := range weights2 {
		weights2[i] = make([]float64, outputSize)
		for j := range weights2[i] {
			weights2[i][j] = rand.Float64()*2 - 1
		}
	}

	bias1 := make([]float64, hiddenSize)
	for i := range bias1 {
		bias1[i] = rand.Float64()*2 - 1
	}

	bias2 := make([]float64, outputSize)
	for i := range bias2 {
		bias2[i] = rand.Float64()*2 - 1
	}

	return &ANN{
		weights1:   weights1,
		weights2:   weights2,
		bias1:      bias1,
		bias2:      bias2,
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
		sum := ann.bias1[i]
		for j := range inputs {
			sum += inputs[j] * ann.weights1[j][i]
		}
		hiddenLayer[i] = sigmoid(sum)
	}

	// Cálculo de la capa de salida
	for i := range outputLayer {
		sum := ann.bias2[i]
		for j := range hiddenLayer {
			sum += hiddenLayer[j] * ann.weights2[j][i]
		}
		outputLayer[i] = sigmoid(sum)
	}

	return outputLayer
}

// Función de entropía cruzada para clasificación binaria
func crossEntropy(yTrue, yPred []float64) float64 {
	sum := 0.0
	for i := range yTrue {
		if yTrue[i] == 1 {
			sum += -math.Log(yPred[i] + 1e-15)
		} else {
			sum += -math.Log(1 - yPred[i] + 1e-15)
		}
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
			sum += outputError[j] * ann.weights2[i][j]
		}
		hiddenLayerError[i] = sigmoidDeriv(sum) * sum
	}

	// Actualizar pesos y sesgos
	for i := 0; i < ann.inputSize; i++ {
		for j := 0; j < ann.hiddenSize; j++ {
			ann.weights1[i][j] += learningRate * hiddenLayerError[j] * inputs[i]
		}
	}

	for i := 0; i < ann.hiddenSize; i++ {
		for j := 0; j < ann.outputSize; j++ {
			ann.weights2[i][j] += learningRate * outputError[j] * hiddenLayerError[i]
		}
	}

	for i := range ann.bias1 {
		ann.bias1[i] += learningRate * hiddenLayerError[i]
	}
	for i := range ann.bias2 {
		ann.bias2[i] += learningRate * outputError[i]
	}
}

// Entrenamiento de la red neuronal
func (ann *ANN) train(data [][]float64, labels []float64, epochs int, learningRate float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		for i := range data {
			output := ann.forward(data[i])
			ann.backpropagate(data[i], []float64{labels[i]}, output, learningRate)
		}
	}
}

// Función de evaluación para entropía cruzada
func evaluate(predictions []float64, labels []float64) (float64, float64, float64) {
	var tp, fp, fn, tn float64
	for i := range predictions {
		pred := predictions[i] > 0.5
		trueLabel := labels[i] > 0.5

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

// Nueva función para ejecutar la red neuronal
func ANNSecuential(data [][]float64, labels []float64) {

	start := time.Now()

	// Definir dimensiones
	inputSize := len(data[0])
	hiddenSize := 5
	outputSize := 1

	// Crear red neuronal
	ann := newANN(inputSize, hiddenSize, outputSize)

	// Entrenar red neuronal
	ann.train(data, labels, 1000, 0.00001)

	// Hacer predicciones
	predictions := make([]float64, len(data))
	for i, d := range data {
		pred := ann.forward(d)[0]
		predictions[i] = pred
	}

	// Evaluar el modelo
	precision, recall, f1 := evaluate(predictions, labels)
	fmt.Printf("Precision: %v\n", precision)
	fmt.Printf("Recall: %v\n", recall)
	fmt.Printf("F1 Score: %v\n", f1)

	elapsed := time.Since(start)
	fmt.Printf("Tiempo de ejecución: %s\n", elapsed)
}
