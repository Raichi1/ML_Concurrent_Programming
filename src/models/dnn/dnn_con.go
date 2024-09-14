package dnn

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
)

// Retropropagación
func (dnn *DNN) backpropagateConcurrent(activations, zs [][]float64, label float64, learningRate float64, wg *sync.WaitGroup) {
	defer wg.Done()

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
func evaluateConcurrent(dnn *DNN, data [][]float64, labels []float64) (float64, float64) {
	var correctPredictions int
	var totalMSE float64
	var wg sync.WaitGroup
	var mu sync.Mutex

	for i := range data {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			outputs, _ := dnn.forward(data[i])
			prediction := math.Round(outputs[len(outputs)-1][0]) // Redondear la salida para obtener 0 o 1
			mu.Lock()
			if prediction == labels[i] {
				correctPredictions++
			}
			totalMSE += costFunction(outputs[len(outputs)-1][0], labels[i])
			mu.Unlock()
		}(i)
	}
	wg.Wait()

	accuracy := float64(correctPredictions) / float64(len(data))
	return accuracy, totalMSE / float64(len(data))
}

// Entrenamiento de la red neuronal profunda
func (dnn *DNN) trainConcurrent(trainData, testData [][]float64, trainLabels, testLabels []float64, epochs int, learningRate float64) {
	for epoch := 0; epoch < epochs; epoch++ {
		totalCost := 0.0
		var wg sync.WaitGroup

		for i := range trainData {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				activations, zs := dnn.forward(trainData[i])
				totalCost += costFunction(activations[len(activations)-1][0], trainLabels[i])
				dnn.backpropagateConcurrent(activations, zs, trainLabels[i], learningRate, &wg)
			}(i)
		}
		wg.Wait()

		trainAccuracy, trainMSE := evaluateConcurrent(dnn, trainData, trainLabels)
		testAccuracy, testMSE := evaluateConcurrent(dnn, testData, testLabels)
		fmt.Printf("Epoch %d: Costo: %f, Precisión entrenamiento: %f, MSE entrenamiento: %f, Precisión prueba: %f, MSE prueba: %f\n",
			epoch, totalCost, trainAccuracy, trainMSE, testAccuracy, testMSE)
	}
}

func DNNConcurrent(train [][]float64, trainLabel []float64, test [][]float64, testLabel []float64) {
	rand.Seed(42)

	// Definir la arquitectura de la red neuronal profunda
	layerSizes := []int{3, 5, 5, 1} // Última capa con 1 neurona para etiquetas como valor único

	// Crear red neuronal profunda
	dnn := newDNN(layerSizes)

	// Entrenar red neuronal profunda
	dnn.trainConcurrent(train, test, trainLabel, testLabel, 1000, 0.0001)

	// // Hacer predicciones
	// for _, d := range test {
	// 	outputs, _ := dnn.forward(d)
	// 	fmt.Println("Predicción:", outputs[len(outputs)-1])
	// }
}
