package randomforest

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

type TreeNode struct {
	FeatureIndex int
	Threshold    float64
	Left, Right  *TreeNode
	Label        int
	IsLeaf       bool
}

type RandomForest struct {
	Trees []*TreeNode
}

// Helper function to create a decision tree
func createTree(data [][]float64, labels []int) *TreeNode {
	if len(data) == 0 {
		return nil
	}

	// If all labels are the same, return a leaf node
	if allSame(labels) {
		return &TreeNode{Label: labels[0], IsLeaf: true}
	}

	// Find the best split
	featureIndex, threshold := findBestSplit(data, labels)
	if featureIndex == -1 {
		return &TreeNode{Label: majorityLabel(labels), IsLeaf: true}
	}

	// Split data
	leftData, leftLabels, rightData, rightLabels := splitData(data, labels, featureIndex, threshold)

	// Create the subtree
	left := createTree(leftData, leftLabels)
	right := createTree(rightData, rightLabels)

	return &TreeNode{
		FeatureIndex: featureIndex,
		Threshold:    threshold,
		Left:         left,
		Right:        right,
	}
}

func allSame(labels []int) bool {
	for i := 1; i < len(labels); i++ {
		if labels[i] != labels[0] {
			return false
		}
	}
	return true
}

func findBestSplit(data [][]float64, labels []int) (int, float64) {
	numFeatures := len(data[0])
	bestFeatureIndex := -1
	bestThreshold := 0.0
	bestScore := math.Inf(-1)

	for featureIndex := 0; featureIndex < numFeatures; featureIndex++ {
		threshold, score := bestThresholdForFeature(data, labels, featureIndex)
		if score > bestScore {
			bestScore = score
			bestFeatureIndex = featureIndex
			bestThreshold = threshold
		}
	}

	return bestFeatureIndex, bestThreshold
}

func bestThresholdForFeature(data [][]float64, labels []int, featureIndex int) (float64, float64) {
	bestThreshold := 0.0
	bestScore := math.Inf(-1)

	values := make(map[float64]bool)
	for _, row := range data {
		values[row[featureIndex]] = true
	}

	for value := range values {
		threshold := value
		leftLabels, rightLabels := splitLabels(data, labels, featureIndex, threshold)
		score := giniIndex(leftLabels, rightLabels)
		if score > bestScore {
			bestScore = score
			bestThreshold = threshold
		}
	}

	return bestThreshold, bestScore
}

func splitData(data [][]float64, labels []int, featureIndex int, threshold float64) ([][]float64, []int, [][]float64, []int) {
	var leftData [][]float64
	var leftLabels []int
	var rightData [][]float64
	var rightLabels []int

	for i, row := range data {
		if row[featureIndex] <= threshold {
			leftData = append(leftData, row)
			leftLabels = append(leftLabels, labels[i])
		} else {
			rightData = append(rightData, row)
			rightLabels = append(rightLabels, labels[i])
		}
	}

	return leftData, leftLabels, rightData, rightLabels
}

func splitLabels(data [][]float64, labels []int, featureIndex int, threshold float64) ([]int, []int) {
	var leftLabels []int
	var rightLabels []int

	for i, row := range data {
		if row[featureIndex] <= threshold {
			leftLabels = append(leftLabels, labels[i])
		} else {
			rightLabels = append(rightLabels, labels[i])
		}
	}

	return leftLabels, rightLabels
}

func giniIndex(leftLabels, rightLabels []int) float64 {
	leftSize := float64(len(leftLabels))
	rightSize := float64(len(rightLabels))
	totalSize := leftSize + rightSize

	if totalSize == 0 {
		return 0
	}

	leftScore := 1 - calculateGini(leftLabels)
	rightScore := 1 - calculateGini(rightLabels)

	return (leftSize/totalSize)*leftScore + (rightSize/totalSize)*rightScore
}

func calculateGini(labels []int) float64 {
	labelCounts := make(map[int]int)
	for _, label := range labels {
		labelCounts[label]++
	}

	size := float64(len(labels))
	gini := 1.0
	for _, count := range labelCounts {
		proportion := float64(count) / size
		gini -= proportion * proportion
	}

	return gini
}

func majorityLabel(labels []int) int {
	labelCounts := make(map[int]int)
	for _, label := range labels {
		labelCounts[label]++
	}

	maxCount := 0
	var majorityLabel int
	for label, count := range labelCounts {
		if count > maxCount {
			maxCount = count
			majorityLabel = label
		}
	}

	return majorityLabel
}

// Train the Random Forest sequentially
func (rf *RandomForest) Train(data [][]float64, labels []int, numTrees int) {
	for i := 0; i < numTrees; i++ {
		sampledData, sampledLabels := bootstrapSample(data, labels)
		tree := createTree(sampledData, sampledLabels)
		rf.Trees = append(rf.Trees, tree)
	}
}

// Predict using the Random Forest
func (rf *RandomForest) Predict(sample []float64) int {
	votes := make(map[int]int)

	for _, tree := range rf.Trees {
		label := predictTree(tree, sample)
		votes[label]++
	}

	var maxVotes int
	var prediction int
	for label, count := range votes {
		if count > maxVotes {
			maxVotes = count
			prediction = label
		}
	}

	return prediction
}

func predictTree(node *TreeNode, sample []float64) int {
	if node.IsLeaf {
		return node.Label
	}

	if sample[node.FeatureIndex] <= node.Threshold {
		return predictTree(node.Left, sample)
	}
	return predictTree(node.Right, sample)
}

// Helper function for bootstrap sampling
func bootstrapSample(data [][]float64, labels []int) ([][]float64, []int) {
	n := len(data)
	sampledData := make([][]float64, n)
	sampledLabels := make([]int, n)
	for i := 0; i < n; i++ {
		index := rand.Intn(n)
		sampledData[i] = data[index]
		sampledLabels[i] = labels[index]
	}
	return sampledData, sampledLabels
}

// Metrics calculation functions

func accuracy(predictions, trueLabels []int) float64 {
	if len(predictions) != len(trueLabels) {
		return 0
	}

	correct := 0
	for i := range predictions {
		if predictions[i] == trueLabels[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(predictions))
}

func precision(predictions, trueLabels []int, positiveClass int) float64 {
	truePos, falsePos := 0, 0
	for i := range predictions {
		if predictions[i] == positiveClass {
			if trueLabels[i] == positiveClass {
				truePos++
			} else {
				falsePos++
			}
		}
	}
	if truePos+falsePos == 0 {
		return 0
	}
	return float64(truePos) / float64(truePos+falsePos)
}

func recall(predictions, trueLabels []int, positiveClass int) float64 {
	truePos, falseNeg := 0, 0
	for i := range predictions {
		if predictions[i] == positiveClass {
			if trueLabels[i] == positiveClass {
				truePos++
			}
		} else {
			if trueLabels[i] == positiveClass {
				falseNeg++
			}
		}
	}
	if truePos+falseNeg == 0 {
		return 0
	}
	return float64(truePos) / float64(truePos+falseNeg)
}

func f1Score(precision, recall float64) float64 {
	if precision+recall == 0 {
		return 0
	}
	return 2 * (precision * recall) / (precision + recall)
}

func RandomForestSecuential(data [][]float64, labels []int, test [][]float64, testLabel []int) {

	start := time.Now()

	rf := &RandomForest{}
	rf.Train(data, labels, 5)

	predictions := make([]int, len(test))

	for i, sample := range test {
		predictions[i] = rf.Predict(sample)
	}

	// Calculate metrics
	acc := accuracy(predictions, testLabel)
	prec := precision(predictions, testLabel, 1)
	rec := recall(predictions, testLabel, 1)
	f1 := f1Score(prec, rec)

	fmt.Printf("Accuracy: %.2f\n", acc)
	fmt.Printf("Precision: %.2f\n", prec)
	fmt.Printf("Recall: %.2f\n", rec)
	fmt.Printf("F1 Score: %.2f\n", f1)

	elapsed := time.Since(start)
	fmt.Printf("Tiempo de ejecución %s\n", elapsed)
}
