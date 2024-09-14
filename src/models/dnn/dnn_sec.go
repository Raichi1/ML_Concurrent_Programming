package dnn

import (
	"fmt"
	"math"
	"math/rand"
)

// Red Neuronal Profunda (DNN)
type DNN struct {
	weights    [][][]float64
	biases     [][]float64
	layerSizes []int
}

// Inicializa la red neuronal profunda
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

// Derivada de la función Sigmoid
func sigmoidDerivative(x float64) float64 {
	return sigmoid(x) * (1 - sigmoid(x))
}

// Propagación hacia adelante
func (dnn *DNN) forward(inputs []float64) ([][]float64, [][]float64) {
	activations := make([][]float64, len(dnn.layerSizes))
	zs := make([][]float64, len(dnn.layerSizes)-1)

	activations[0] = inputs
	for i := 0; i < len(dnn.weights); i++ {
		layerOutput := make([]float64, dnn.layerSizes[i+1])
		zs[i] = make([]float64, dnn.layerSizes[i+1])
		for j := 0; j < dnn.layerSizes[i+1]; j++ {
			sum := dnn.biases[i][j]
			for k := 0; k < dnn.layerSizes[i]; k++ {
				sum += activations[i][k] * dnn.weights[i][k][j]
			}
			zs[i][j] = sum
			layerOutput[j] = sigmoid(sum)
		}
		activations[i+1] = layerOutput
	}
	return activations, zs
}

// Función de costo (MSE)
func costFunction(output float64, label float64) float64 {
	return 0.5 * math.Pow(output-label, 2)
}

// Derivada de la función de costo respecto a la salida
func costDerivative(output float64, label float64) float64 {
	return output - label
}

// Retropropagación
func (dnn *DNN) backpropagate(activations, zs [][]float64, label float64, learningRate float64) {
	// Inicializar los gradientes
	weightGradients := make([][][]float64, len(dnn.weights))
	biasGradients := make([][]float64, len(dnn.biases))

	// Derivada del costo respecto a la última capa
	delta := make([]float64, dnn.layerSizes[len(dnn.layerSizes)-1])
	for j := range delta {
		delta[j] = costDerivative(activations[len(activations)-1][j], label) * sigmoidDerivative(zs[len(zs)-1][j])
	}

	// Actualizar los gradientes para la última capa
	biasGradients[len(biasGradients)-1] = delta
	weightGradients[len(weightGradients)-1] = make([][]float64, dnn.layerSizes[len(dnn.layerSizes)-2])
	for j := range weightGradients[len(weightGradients)-1] {
		weightGradients[len(weightGradients)-1][j] = make([]float64, dnn.layerSizes[len(dnn.layerSizes)-1])
		for k := range weightGradients[len(weightGradients)-1][j] {
			weightGradients[len(weightGradients)-1][j][k] = activations[len(activations)-2][j] * delta[k]
		}
	}

	// Retropropagación para capas anteriores
	for l := len(zs) - 2; l >= 0; l-- {
		nextDelta := make([]float64, dnn.layerSizes[l+1])
		for j := range nextDelta {
			sum := 0.0
			for k := range delta {
				sum += dnn.weights[l+1][j][k] * delta[k]
			}
			nextDelta[j] = sum * sigmoidDerivative(zs[l][j])
		}
		delta = nextDelta

		biasGradients[l] = delta
		weightGradients[l] = make([][]float64, dnn.layerSizes[l])
		for j := range weightGradients[l] {
			weightGradients[l][j] = make([]float64, dnn.layerSizes[l+1])
			for k := range weightGradients[l][j] {
				weightGradients[l][j][k] = activations[l][j] * delta[k]
			}
		}
	}

	// Actualizar pesos y sesgos
	for i := range dnn.weights {
		for j := range dnn.weights[i] {
			for k := range dnn.weights[i][j] {
				dnn.weights[i][j][k] -= learningRate * weightGradients[i][j][k]
			}
		}
		for j := range dnn.biases[i] {
			dnn.biases[i][j] -= learningRate * biasGradients[i][j]
		}
	}
}

// Métricas de evaluación: precisión y MSE
func evaluate(dnn *DNN, data [][]float64, labels []float64) (float64, float64) {
	var correctPredictions int
	var totalMSE float64

	for i := range data {
		outputs, _ := dnn.forward(data[i])
		prediction := math.Round(outputs[len(outputs)-1][0]) // Redondear la salida para obtener 0 o 1
		if prediction == labels[i] {
			correctPredictions++
		}
		totalMSE += costFunction(outputs[len(outputs)-1][0], labels[i])
	}

	accuracy := float64(correctPredictions) / float64(len(data))
	return accuracy, totalMSE / float64(len(data))
}

// Entrenamiento de la red neuronal profunda
func (dnn *DNN) train(trainData, testData [][]float64, trainLabels, testLabels []float64, epochs int, learningRate float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		totalCost := 0.0
		for i := range trainData {
			activations, zs := dnn.forward(trainData[i])
			totalCost += costFunction(activations[len(activations)-1][0], trainLabels[i])
			dnn.backpropagate(activations, zs, trainLabels[i], learningRate)
		}
		trainAccuracy, trainMSE := evaluate(dnn, trainData, trainLabels)
		testAccuracy, testMSE := evaluate(dnn, testData, testLabels)
		fmt.Printf("Epoch %d: Costo: %f, Precisión entrenamiento: %f, MSE entrenamiento: %f, Precisión prueba: %f, MSE prueba: %f\n",
			epoch, totalCost, trainAccuracy, trainMSE, testAccuracy, testMSE)
	}
}

func DNNSecuential(train [][]float64, trainLabel []float64, test [][]float64, testLabel []float64) {
	rand.Seed(42)

	// Definir la arquitectura de la red neuronal profunda
	layerSizes := []int{3, 5, 5, 1} // Última capa con 1 neurona para etiquetas como valor único

	// Crear red neuronal profunda
	dnn := newDNN(layerSizes)

	// Datos de ejemplo (cada fila tiene 3 características y las etiquetas son una lista unidimensional)
	// data := [][]float64{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9}, {0.9, 0.7, 0.5}}
	// labels := []float64{0.0, 1.0, 1.0, 0.0} // Etiquetas como una lista unidimensional

	// Dividir datos en entrenamiento y prueba (50% entrenamiento, 50% prueba)
	// trainData := data[:2]
	// testData := data[2:]
	// trainLabels := labels[:2]
	// testLabels := labels[2:]

	// Entrenar red neuronal profunda
	dnn.train(train, test, trainLabel, testLabel, 1000, 0.0001)

	// // Hacer predicciones
	// for _, d := range test {
	// 	outputs, _ := dnn.forward(d)
	// 	fmt.Println("Predicción:", outputs[len(outputs)-1])
	// }
}
