package decisiontree

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// Función para dividir los datos basada en el feature y threshold de manera concurrente
func splitDataConcurrente(data [][]float64, labels []float64, feature int, threshold float64) ([][]float64, [][]float64, []float64, []float64) {
	var leftData, rightData [][]float64
	var leftLabels, rightLabels []float64

	for i, point := range data {
		if point[feature] <= threshold {
			leftData = append(leftData, point)
			leftLabels = append(leftLabels, labels[i])
		} else {
			rightData = append(rightData, point)
			rightLabels = append(rightLabels, labels[i])
		}
	}

	return leftData, rightData, leftLabels, rightLabels
}

// Función para entrenar el árbol de decisión concurrentemente
func trainDecisionTreeConcurrente(data [][]float64, labels []float64, depth int) *Node {
	if depth == 0 || len(data) == 0 {
		return &Node{Prediction: mean(labels)}
	}

	bestFeature, bestThreshold := findBestSplitConcurrente(data, labels)
	if bestFeature == -1 {
		return &Node{Prediction: mean(labels)}
	}

	leftData, rightData, leftLabels, rightLabels := splitDataConcurrente(data, labels, bestFeature, bestThreshold)

	if len(leftData) == 0 || len(rightData) == 0 {
		return &Node{Prediction: mean(labels)}
	}

	// Uso de goroutines para entrenar los subárboles izquierdo y derecho en paralelo
	var wg sync.WaitGroup
	wg.Add(2)

	var leftNode, rightNode *Node

	go func() {
		defer wg.Done()
		leftNode = trainDecisionTreeConcurrente(leftData, leftLabels, depth-1)
	}()

	go func() {
		defer wg.Done()
		rightNode = trainDecisionTreeConcurrente(rightData, rightLabels, depth-1)
	}()

	wg.Wait()

	return &Node{
		Feature:   bestFeature,
		Threshold: bestThreshold,
		Left:      leftNode,
		Right:     rightNode,
	}
}

// Función para encontrar la mejor división concurrentemente
func findBestSplitConcurrente(data [][]float64, labels []float64) (int, float64) {
	bestFeature := -1
	bestThreshold := 0.0
	bestImpurity := math.Inf(1)
	numFeatures := len(data[0])

	// Canal para compartir resultados de las divisiones
	type SplitResult struct {
		feature       int
		threshold     float64
		totalImpurity float64
	}

	resultChan := make(chan SplitResult, numFeatures)

	// Función que calcula la mejor división para un feature dado
	var wg sync.WaitGroup
	featureWorker := func(feature int) {
		defer wg.Done()

		uniqueValues := make(map[float64]bool)
		for _, point := range data {
			uniqueValues[point[feature]] = true
		}

		bestFeatureImpurity := math.Inf(1)
		bestFeatureThreshold := 0.0

		for threshold := range uniqueValues {
			_, _, leftLabels, rightLabels := splitDataConcurrente(data, labels, feature, threshold)
			if len(leftLabels) == 0 || len(rightLabels) == 0 {
				continue
			}

			leftImpurity := giniImpurity(leftLabels)
			rightImpurity := giniImpurity(rightLabels)
			totalImpurity := (leftImpurity*float64(len(leftLabels)) + rightImpurity*float64(len(rightLabels))) / float64(len(labels))

			if totalImpurity < bestFeatureImpurity {
				bestFeatureImpurity = totalImpurity
				bestFeatureThreshold = threshold
			}
		}

		// Enviar el resultado a través del canal
		resultChan <- SplitResult{feature, bestFeatureThreshold, bestFeatureImpurity}
	}

	wg.Add(numFeatures)

	for feature := 0; feature < numFeatures; feature++ {
		go featureWorker(feature)
	}

	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Encontrar el mejor resultado de las goroutines
	for result := range resultChan {
		if result.totalImpurity < bestImpurity {
			bestImpurity = result.totalImpurity
			bestFeature = result.feature
			bestThreshold = result.threshold
		}
	}

	return bestFeature, bestThreshold
}

// Función para hacer predicciones (sin cambios)
func predictConcurrente(node *Node, point []float64) float64 {
	if node.Left == nil && node.Right == nil {
		return node.Prediction
	}

	if point[node.Feature] <= node.Threshold {
		return predictConcurrente(node.Left, point)
	} else {
		return predictConcurrente(node.Right, point)
	}
}

// Función para evaluar el rendimiento del árbol concurrentemente
func evaluateConcurrente(data [][]float64, labels []float64, tree *Node) {
	var tp, fp, tn, fn int

	for i, point := range data {
		predLabel := math.Round(predictConcurrente(tree, point))

		if predLabel == 1.0 && labels[i] == 1.0 {
			tp++
		} else if predLabel == 1.0 && labels[i] == 0.0 {
			fp++
		} else if predLabel == 0.0 && labels[i] == 0.0 {
			tn++
		} else if predLabel == 0.0 && labels[i] == 1.0 {
			fn++
		}
	}

	fmt.Printf("TP: %d, FP: %d, TN: %d, FN: %d\n", tp, fp, tn, fn)

	// Calcular precisión
	precision := precisionScore(tp, fp)
	fmt.Printf("Precisión: %.2f\n", precision)
}

// Función para calcular la precisión
func precisionScore(tp, fp int) float64 {
	if tp+fp == 0 {
		return 0
	}
	return float64(tp) / float64(tp+fp)
}

// Función principal para el árbol de decisión concurrente
func DecisionTreeConcurrente(data [][]float64, labels []float64) {
	start := time.Now()

	tree := trainDecisionTreeConcurrente(data, labels, 3)
	evaluateConcurrente(data, labels, tree)

	elapsed := time.Since(start)
	fmt.Printf("Tiempo de ejecución: %s\n", elapsed)
}
