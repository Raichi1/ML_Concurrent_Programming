package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

func generateData(numInputs int, numWeights int) ([]float64, [][]float64) {
	// rand.Seed(time.Now().UnixNano()) // Inicializa la semilla aleatoria

	// Genera los valores para los inputs
	inputs := make([]float64, numInputs)
	for i := range inputs {
		inputs[i] = rand.Float64() // Genera un número aleatorio entre 0 y 1
	}

	// Genera la matriz de pesos
	weights := make([][]float64, numInputs)
	for i := range weights {
		weights[i] = make([]float64, numWeights)
		for j := range weights[i] {
			weights[i][j] = rand.Float64() // Genera un número aleatorio entre 0 y 1
		}
	}

	return inputs, weights
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func forwardPass(weights [][]float64, inputs []float64) float64 {
	sum := 0.0

	for i := 0; i < len(weights); i++ {
		for j := 0; j < len(weights[i]); j++ {
			sum += weights[i][j] * inputs[j]
		}
	}

	return sigmoid(sum)
}

func main() {
	// Captura el tiempo de inicio
	start := time.Now()

	// Definir las dimensiones deseadas
	numInputs := 5
	numWeights := 5

	// Generar los datos
	inputs, weights := generateData(numInputs, numWeights)

	output := forwardPass(weights, inputs)
	fmt.Println("Output:", output)

	// Captura el tiempo de finalización
	elapsed := time.Since(start)

	fmt.Printf("Time taken: %s\n", elapsed)
}
