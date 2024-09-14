package randomforest

import (
	"fmt"
	"sync"
	"time"
)

type RandomForestConc struct {
	Trees []*TreeNode
	mu    sync.Mutex
}

// Train the Random Forest concurrently
func (rf *RandomForestConc) Train(data [][]float64, labels []int, numTrees int) {
	var wg sync.WaitGroup
	rf.mu = sync.Mutex{}

	for i := 0; i < numTrees; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			sampledData, sampledLabels := bootstrapSample(data, labels)
			tree := createTree(sampledData, sampledLabels)
			rf.mu.Lock()
			rf.Trees = append(rf.Trees, tree)
			rf.mu.Unlock()
		}()
	}
	wg.Wait()
}

// Predict using the Random Forest
func (rf *RandomForestConc) Predict(sample []float64) int {
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

func RandomForestConcurrent(data [][]float64, labels []int, test [][]float64, testLabel []int) {

	start := time.Now()

	rf := &RandomForestConc{}
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
	fmt.Printf("Tiempo de ejecución: %s\n", elapsed)
}
