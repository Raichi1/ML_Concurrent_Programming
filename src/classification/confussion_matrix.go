package classification

import "sync"

// ConfusionMatrix genera una matriz de confusión
func ConfusionMatrix(predictions []int, actuals []int) (tp, tn, fp, fn int) {
	for i := range predictions {
		if predictions[i] == 1 && actuals[i] == 1 {
			tp++
		} else if predictions[i] == 0 && actuals[i] == 0 {
			tn++
		} else if predictions[i] == 1 && actuals[i] == 0 {
			fp++
		} else if predictions[i] == 0 && actuals[i] == 1 {
			fn++
		}
	}
	return tp, tn, fp, fn
}

func confusionMatrixConcurrent(actual []int, predicted []int) (tp, fp, tn, fn int) {
	var wg sync.WaitGroup
	var mu sync.Mutex

	numGoroutines := 4 // Número de goroutines (ajuste según CPU)
	chunkSize := len(actual) / numGoroutines

	for i := 0; i < numGoroutines; i++ {
		startIdx := i * chunkSize
		endIdx := startIdx + chunkSize
		if i == numGoroutines-1 {
			endIdx = len(actual)
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			localTP, localFP, localTN, localFN := 0, 0, 0, 0
			for j := start; j < end; j++ {
				if actual[j] == 1 && predicted[j] == 1 {
					localTP++
				} else if actual[j] == 0 && predicted[j] == 1 {
					localFP++
				} else if actual[j] == 0 && predicted[j] == 0 {
					localTN++
				} else if actual[j] == 1 && predicted[j] == 0 {
					localFN++
				}
			}
			mu.Lock()
			tp += localTP
			fp += localFP
			tn += localTN
			fn += localFN
			mu.Unlock()
		}(startIdx, endIdx)
	}

	wg.Wait()
	return
}
