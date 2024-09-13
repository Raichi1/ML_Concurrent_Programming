package colaborativefilter

import (
	"math"
	"sync"
)

// Función para predecir la calificación de un usuario para un ítem
func predictRatingC(userIndex, itemIndex int, ratings [][]float64, similarities []float64, wg *sync.WaitGroup, ch chan float64) {
	defer wg.Done()

	var numerator, denominator float64
	for i, sim := range similarities {
		if i != userIndex && ratings[i][itemIndex] > 0 {
			numerator += sim * ratings[i][itemIndex]
			denominator += math.Abs(sim)
		}
	}

	if denominator == 0 {
		ch <- 0 // Si no hay denominador, retornar 0
		return
	}

	ch <- numerator / denominator
}

// Función pública para obtener recomendaciones
func GetRecommendationsC(ratings [][]float64, userIndex int, itemIndices []int) []float64 {
	numUsers := len(ratings)
	similarityMatrix := make([][]float64, numUsers)
	for i := range similarityMatrix {
		similarityMatrix[i] = make([]float64, numUsers)
	}

	// Calcular la similitud entre usuarios en paralelo
	var wg sync.WaitGroup
	for i := 0; i < numUsers; i++ {
		for j := i + 1; j < numUsers; j++ {
			wg.Add(1)
			go func(i, j int) {
				defer wg.Done()
				similarity := cosineSimilarity(ratings[i], ratings[j])
				similarityMatrix[i][j] = similarity
				similarityMatrix[j][i] = similarity
			}(i, j)
		}
	}
	wg.Wait()

	// Predecir las calificaciones para el usuario especificado
	ch := make(chan float64, len(itemIndices))
	wg.Add(len(itemIndices))
	for _, itemIndex := range itemIndices {
		go predictRatingC(userIndex, itemIndex, ratings, similarityMatrix[userIndex], &wg, ch)
	}
	wg.Wait()
	close(ch)

	// Recoger las predicciones
	var recommendations []float64
	for rating := range ch {
		recommendations = append(recommendations, rating)
	}

	return recommendations
}
