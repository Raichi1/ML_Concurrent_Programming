package main

import (
	"fmt"
	"math"
	"sync"
)

// Define the structure for the Decision Tree Node
type TreeNode struct {
	Feature   int
	Threshold float64
	Left      *TreeNode
	Right     *TreeNode
	Label     float64
	IsLeaf    bool
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

// Function to calculate Gini impurity
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

// Function to find the best split (concurrent version)
func findBestSplit(data [][]float64, labels []float64) (int, float64) {
	bestFeature := -1
	bestThreshold := 0.0
	bestImpurity := math.Inf(1)
	var mu sync.Mutex

	var wg sync.WaitGroup
	numFeatures := len(data[0])
	for feature := 0; feature < numFeatures; feature++ {
		wg.Add(1)
		go func(feature int) {
			defer wg.Done()
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

				mu.Lock()
				if totalImpurity < bestImpurity {
					bestImpurity = totalImpurity
					bestFeature = feature
					bestThreshold = threshold
				}
				mu.Unlock()
			}
		}(feature)
	}

	wg.Wait()
	return bestFeature, bestThreshold
}

// Function to predict labels using the decision tree
func predict(tree *TreeNode, point []float64) float64 {
	if tree.IsLeaf {
		return tree.Label
	}
	if point[tree.Feature] <= tree.Threshold {
		return predict(tree.Left, point)
	}
	return predict(tree.Right, point)
}

// Function to calculate precision
func precision(tp, fp int) float64 {
	if tp+fp == 0 {
		return 0
	}
	return float64(tp) / float64(tp+fp)
}

// Function to calculate recall
func recall(tp, fn int) float64 {
	if tp+fn == 0 {
		return 0
	}
	return float64(tp) / float64(tp+fn)
}

// Function to calculate F1-score
func f1Score(precision, recall float64) float64 {
	if precision+recall == 0 {
		return 0
	}
	return 2 * (precision * recall) / (precision + recall)
}

// Función para calcular la accuracy
func accuracy(tp, tn, fp, fn int) float64 {
	total := tp + tn + fp + fn
	if total == 0 {
		return 0.0 // Evita la división entre 0
	}

	return float64(tp+tn) / float64(total)
}

// Function to evaluate the model and write metrics to CSV
func evaluate(data [][]float64, trueLabels []float64, tree *TreeNode) map[string]float64 {
	var tp, fp, fn, tn int

	for i, point := range data {
		predictedLabel := predict(tree, point)
		actualLabel := trueLabels[i]

		if predictedLabel == actualLabel {
			if predictedLabel == 1 {
				tp++
			} else {
				tn++
			}
		} else {
			if predictedLabel == 1 {
				fp++
			} else {
				fn++
			}
		}
	}

	precisionVal := precision(tp, fp)
	recallVal := recall(tp, fn)
	f1ScoreVal := f1Score(precisionVal, recallVal)

	metrics := map[string]float64{
		"Precision": precisionVal,
		"Recall":    recallVal,
		"F1-Score":  f1ScoreVal,
	}

	return metrics
}

func main() {
	// Example data and labels
	data := [][]float64{
		{1.0, 2.0},
		{2.0, 3.0},
		{3.0, 1.0},
	}
	labels := []float64{1, 1, 0}

	// Build the tree
	tree := &TreeNode{
		Feature:   0,
		Threshold: 2.0,
		Left: &TreeNode{
			Label:  1,
			IsLeaf: true,
		},
		Right: &TreeNode{
			Label:  0,
			IsLeaf: true,
		},
	}

	// Find the best split
	bestFeature, bestThreshold := findBestSplit(data, labels)
	fmt.Printf("Best Feature: %d, Best Threshold: %.2f\n", bestFeature, bestThreshold)

	// Evaluate the model
	metrics := evaluate(data, labels, tree)

	for metric, value := range metrics {
		fmt.Printf("%s: %f \n", metric, value)
	}
}
