package colaborativefilter

import (
	"math"
)

// Función para calcular la similitud del coseno entre dos usuarios
func cosineSimilarity(userA, userB []float64) float64 {
	var dotProduct, normA, normB float64
	for i := range userA {
		dotProduct += userA[i] * userB[i]
		normA += userA[i] * userA[i]
		normB += userB[i] * userB[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

// Función pública para obtener recomendaciones sin concurrencia
func GetRecommendations(ratings [][]float64, userIndex int, itemIndices []int) []float64 {
	numUsers := len(ratings)
	similarityMatrix := make([][]float64, numUsers)
	for i := range similarityMatrix {
		similarityMatrix[i] = make([]float64, numUsers)
	}

	// Calcular la similitud entre usuarios de forma secuencial
	for i := 0; i < numUsers; i++ {
		for j := i + 1; j < numUsers; j++ {
			similarity := cosineSimilarity(ratings[i], ratings[j])
			similarityMatrix[i][j] = similarity
			similarityMatrix[j][i] = similarity
		}
	}

	// Predecir las calificaciones para el usuario especificado de forma secuencial
	var recommendations []float64
	for _, itemIndex := range itemIndices {
		rating := predictRating(userIndex, itemIndex, ratings, similarityMatrix[userIndex])
		recommendations = append(recommendations, rating)
	}

	return recommendations
}

// Función para predecir la calificación de un usuario para un ítem
func predictRating(userIndex, itemIndex int, ratings [][]float64, similarities []float64) float64 {
	var numerator, denominator float64
	for i, sim := range similarities {
		if i != userIndex && ratings[i][itemIndex] > 0 {
			numerator += sim * ratings[i][itemIndex]
			denominator += math.Abs(sim)
		}
	}

	if denominator == 0 {
		return 0
	}

	return numerator / denominator
}
