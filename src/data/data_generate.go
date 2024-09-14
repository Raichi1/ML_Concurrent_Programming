package data

// import (
// 	"math"
// 	"math/rand"
// 	"sync"
// )

// func generateData(size int) ([]int, []int) {
// 	actual := make([]int, size)
// 	predicted := make([]int, size)
// 	for i := 0; i < size; i++ {
// 		actual[i] = rand.Intn(2) // Genera 0 o 1
// 		predicted[i] = rand.Intn(2)
// 	}
// 	return actual, predicted
// }

// // Función para calcular la impureza Gini
// func giniImpurity(labels []float64) float64 {
// 	labelCounts := make(map[float64]int)

// 	for _, label := range labels {
// 		labelCounts[label]++
// 	}

// 	impurity := 1.0
// 	total := float64(len(labels))

// 	for _, count := range labelCounts {
// 		probability := float64(count) / total
// 		impurity -= probability * probability
// 	}

// 	return impurity
// }

// func entropy(labels []float64) float64 {
// 	labelCounts := make(map[float64]int)
// 	for _, label := range labels {
// 		labelCounts[label]++
// 	}

// 	entropy := 0.0
// 	total := float64(len(labels))
// 	for _, count := range labelCounts {
// 		probability := float64(count) / total
// 		entropy -= probability * math.Log2(probability)
// 	}

// 	return entropy
// }

// // Función para encontrar la mejor división
// func findBestSplit(data [][]float64, labels []float64) (int, float64) {
// 	bestFeature := -1
// 	bestThreshold := 0.0
// 	bestImpurity := math.Inf(1)

// 	for feature := range data[0] {
// 		uniqueValues := make(map[float64]bool)
// 		for _, point := range data {
// 			uniqueValues[point[feature]] = true
// 		}

// 		for threshold := range uniqueValues {
// 			_, _, leftLabels, rightLabels := splitData(data, labels, feature, threshold)
// 			if len(leftLabels) == 0 || len(rightLabels) == 0 {
// 				continue
// 			}

// 			leftImpurity := giniImpurity(leftLabels)
// 			rightImpurity := giniImpurity(rightLabels)
// 			totalImpurity := (leftImpurity*float64(len(leftLabels)) + rightImpurity*float64(len(rightLabels))) / float64(len(labels))

// 			if totalImpurity < bestImpurity {
// 				bestImpurity = totalImpurity
// 				bestFeature = feature
// 				bestThreshold = threshold
// 			}
// 		}
// 	}

// 	return bestFeature, bestThreshold
// }

// func findBestSplitConcurrent(data [][]float64, labels []float64) (int, float64) {
// 	bestFeature := -1
// 	bestThreshold := 0.0
// 	bestImpurity := math.Inf(1)
// 	var mu sync.Mutex
// 	var wg sync.WaitGroup

// 	for feature := range data[0] {
// 		// Crear un mapa de valores únicos para la característica
// 		uniqueValues := make(map[float64]bool)
// 		for _, point := range data {
// 			uniqueValues[point[feature]] = true
// 		}

// 		// Para cada valor único, evaluar la división
// 		for threshold := range uniqueValues {
// 			wg.Add(1)
// 			go func(feature int, threshold float64) {
// 				defer wg.Done()
// 				// Obtener subconjuntos de datos con base en la característica y el umbral
// 				leftData, rightData, leftLabels, rightLabels := splitData(data, labels, feature, threshold)

// 				// Si alguna división está vacía, no es válida
// 				if len(leftLabels) == 0 || len(rightLabels) == 0 {
// 					return
// 				}

// 				// Calcular las impurezas de Gini para los subconjuntos
// 				leftImpurity := giniImpurity(leftLabels)
// 				rightImpurity := giniImpurity(rightLabels)
// 				totalImpurity := (leftImpurity*float64(len(leftLabels)) + rightImpurity*float64(len(rightLabels))) / float64(len(labels))

// 				// Bloquear el acceso a las variables compartidas para evitar condiciones de carrera
// 				mu.Lock()
// 				defer mu.Unlock()
// 				if totalImpurity < bestImpurity {
// 					bestImpurity = totalImpurity
// 					bestFeature = feature
// 					bestThreshold = threshold
// 				}

// 			}(feature, threshold)
// 		}
// 	}

// 	// Esperar a que todas las goroutines terminen
// 	wg.Wait()
// 	return bestFeature, bestThreshold
// }
