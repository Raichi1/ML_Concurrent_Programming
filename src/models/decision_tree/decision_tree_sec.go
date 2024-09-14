package decisiontree

import (
	"fmt"
	"math"
	"time"
)

// Nodo del árbol de decisión
type Node struct {
	Feature    int
	Threshold  float64
	Left       *Node
	Right      *Node
	Prediction float64
}

// Function to split data based on feature and threshold
func splitData(data [][]float64, labels []float64, feature int, threshold float64) ([][]float64, [][]float64, []float64, []float64) {
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

// Función para entrenar el árbol de decisión
func trainDecisionTree(data [][]float64, labels []float64, depth int) *Node {
	if depth == 0 || len(data) == 0 {
		return &Node{Prediction: mean(labels)}
	}

	bestFeature, bestThreshold := findBestSplit(data, labels)
	if bestFeature == -1 {
		return &Node{Prediction: mean(labels)}
	}

	leftData, rightData, leftLabels, rightLabels := splitData(data, labels, bestFeature, bestThreshold)

	if len(leftData) == 0 || len(rightData) == 0 {
		return &Node{Prediction: mean(labels)}
	}

	// fmt.Printf("División: Feature %d, Threshold %.2f\n", bestFeature, bestThreshold)

	return &Node{
		Feature:   bestFeature,
		Threshold: bestThreshold,
		Left:      trainDecisionTree(leftData, leftLabels, depth-1),
		Right:     trainDecisionTree(rightData, rightLabels, depth-1),
	}
}

// Función para encontrar la mejor división
func findBestSplit(data [][]float64, labels []float64) (int, float64) {
	bestFeature := -1
	bestThreshold := 0.0
	bestImpurity := math.Inf(1)
	numFeatures := len(data[0])

	for feature := 0; feature < numFeatures; feature++ {
		uniqueValues := make(map[float64]bool)
		for _, point := range data {
			uniqueValues[point[feature]] = true
		}

		for threshold := range uniqueValues {
			_, _, leftLabels, rightLabels := splitData(data, labels, feature, threshold)
			if len(leftLabels) == 0 || len(rightLabels) == 0 {
				continue
			}

			leftImpurity := giniImpurity(leftLabels)
			rightImpurity := giniImpurity(rightLabels)
			totalImpurity := (leftImpurity*float64(len(leftLabels)) + rightImpurity*float64(len(rightLabels))) / float64(len(labels))

			if totalImpurity < bestImpurity {
				bestImpurity = totalImpurity
				bestFeature = feature
				bestThreshold = threshold
			}
		}
	}

	return bestFeature, bestThreshold
}

// Función para calcular la impureza Gini
func giniImpurity(labels []float64) float64 {
	labelCounts := make(map[float64]int)
	for _, label := range labels {
		labelCounts[label]++
	}

	impurity := 1.0
	total := float64(len(labels))
	for _, count := range labelCounts {
		probability := float64(count) / total
		impurity -= probability * probability
	}

	return impurity
}

// Function to calculate precision
func precision(tp, fp int) float64 {
	if tp+fp == 0 {
		return 0
	}
	return float64(tp) / float64(tp+fp)
}

// Función para calcular la media
func mean(labels []float64) float64 {
	if len(labels) == 0 {
		return 0
	}
	sum := 0.0
	for _, label := range labels {
		sum += label
	}
	return sum / float64(len(labels))
}

// Función para hacer predicciones con el árbol entrenado
func predict(node *Node, point []float64) float64 {
	if node.Left == nil && node.Right == nil {
		return node.Prediction
	}

	if point[node.Feature] <= node.Threshold {
		return predict(node.Left, point)
	} else {
		return predict(node.Right, point)
	}
}

// Función para evaluar el rendimiento del árbol con las métricas: Precisión, Recall, F1-Score, Accuracy
func evaluate(data [][]float64, labels []float64, tree *Node) {
	var tp, fp, tn, fn int // Verdaderos positivos, falsos positivos, verdaderos negativos, falsos negativos

	for i, point := range data {
		predLabel := math.Round(predict(tree, point))

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

	// Calcular precisión, recall, F1-score y exactitud
	precision := precision(tp, fp)
	// recall := recall(tp, fn)
	// f1Score := f1Score(precision, recall)
	// accuracy := accuracy(tp, tn, fp, fn)

	fmt.Printf("Precisión: %.2f\n", precision)
}

func DecisionTreeSec(data [][]float64, labels []float64) {
	// Tiempo inicial
	start := time.Now()

	// Datos de ejemplo
	// data := [][]float64{{2.3, 4.5}, {1.3, 3.5}, {3.3, 2.5}, {2.5, 3.8}, {1.9, 2.8}}
	// labels := []float64{1.0, 0.0, 1.0, 0.0, 1.0}

	// Entrenar el árbol de decisión
	tree := trainDecisionTree(data, labels, 3)

	// Evaluar el rendimiento del árbol
	evaluate(data, labels, tree)

	// Tiempo transcurrido
	elapsed := time.Since(start)
	fmt.Printf("Tiempo de ejecución: %s\n", elapsed)
}
