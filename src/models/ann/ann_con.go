package ann

import (
	"fmt"
	"sync"
	"time"
)

// Propagación hacia adelante
func (ann *ANN) forwardConcurrent(inputs []float64) []float64 {
	hiddenLayer := make([]float64, ann.hiddenSize)
	outputLayer := make([]float64, ann.outputSize)

	var wg sync.WaitGroup
	wg.Add(2)

	// Cálculo de la capa oculta
	go func() {
		defer wg.Done()
		for i := range hiddenLayer {
			sum := ann.bias1[i]
			for j := range inputs {
				sum += inputs[j] * ann.weights1[j][i]
			}
			hiddenLayer[i] = sigmoid(sum)
		}
	}()

	// Cálculo de la capa de salida
	go func() {
		defer wg.Done()
		for i := range outputLayer {
			sum := ann.bias2[i]
			for j := range hiddenLayer {
				sum += hiddenLayer[j] * ann.weights2[j][i]
			}
			outputLayer[i] = sigmoid(sum)
		}
	}()

	wg.Wait()
	return outputLayer
}

// Ajuste de pesos y sesgos (Backpropagation)
func (ann *ANN) backpropagateConcurrent(inputs []float64, labels []float64, output []float64, learningRate float64) {
	// Calcular el error de la capa de salida
	outputError := make([]float64, ann.outputSize)
	for i := range outputError {
		outputError[i] = labels[i] - output[i]
	}

	// Calcular el error de la capa oculta
	hiddenLayerError := make([]float64, ann.hiddenSize)
	var wg sync.WaitGroup
	wg.Add(2)

	// Error de la capa oculta
	go func() {
		defer wg.Done()
		for i := range hiddenLayerError {
			sum := 0.0
			for j := range outputError {
				sum += outputError[j] * ann.weights2[i][j]
			}
			hiddenLayerError[i] = sigmoidDeriv(sum) * sum
		}
	}()

	// Actualizar pesos y sesgos
	go func() {
		defer wg.Done()
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
	}()

	wg.Wait()
}

// Entrenamiento de la red neuronal
func (ann *ANN) trainConcurrent(data [][]float64, labels []float64, epochs int, learningRate float64) {
	var wg sync.WaitGroup
	for epoch := 0; epoch < epochs; epoch++ {
		wg.Add(len(data))
		for i := range data {
			go func(i int) {
				defer wg.Done()
				output := ann.forwardConcurrent(data[i])
				ann.backpropagateConcurrent(data[i], []float64{labels[i]}, output, learningRate)
			}(i)
		}
		wg.Wait()
	}
}

// Nueva función para ejecutar la red neuronal
func ANNConcurrent(data [][]float64, labels []float64) {
	start := time.Now()

	// Definir dimensiones
	inputSize := len(data[0])
	hiddenSize := 5
	outputSize := 1

	// Crear red neuronal
	ann := newANN(inputSize, hiddenSize, outputSize)

	// Entrenar red neuronal
	ann.trainConcurrent(data, labels, 1000, 0.00001)

	// Hacer predicciones
	predictions := make([]float64, len(data))
	var wg sync.WaitGroup
	wg.Add(len(data))
	for i, d := range data {
		go func(i int, d []float64) {
			defer wg.Done()
			pred := ann.forward(d)[0]
			predictions[i] = pred
		}(i, d)
	}
	wg.Wait()

	// Evaluar el modelo
	precision, recall, f1 := evaluate(predictions, labels)
	fmt.Printf("Precision: %v\n", precision)
	fmt.Printf("Recall: %v\n", recall)
	fmt.Printf("F1 Score: %v\n", f1)

	elapsed := time.Since(start)
	fmt.Printf("Tiempo de ejecución: %s\n", elapsed)
}
